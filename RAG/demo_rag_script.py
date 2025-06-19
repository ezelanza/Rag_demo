#!/usr/bin/env python3
"""
RAG Demo Script - Complete Implementation
=========================================

This script replicates the functionality of Demo.ipynb notebook, providing:
- Automatic model download (GGUF format for optimal performance)
- PDF document processing and vector store creation
- RAG pipeline with llama-cpp-python integration
- Comparison between direct LLM and RAG responses
- Interactive and batch query modes

Usage:
    python demo_rag_script.py --setup                    # Initial setup only
    python demo_rag_script.py --query "What is DRAGON?" # Single query
    python demo_rag_script.py --interactive             # Interactive mode
    python demo_rag_script.py --compare --query "..."   # Compare direct vs RAG
    python demo_rag_script.py --batch queries.txt       # Batch processing
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

# Core imports
try:
    from huggingface_hub import login, hf_hub_download
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_community.llms import LlamaCpp
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("📝 Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Check if .env file exists, create it if it doesn't
def ensure_env_file():
    """Ensure .env file exists with template content"""
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# RAG Demo Environment Variables\n")
            f.write("# Replace the placeholder with your actual Hugging Face token\n")
            f.write("# Get your token from: https://huggingface.co/settings/tokens\n")
            f.write("HUGGINGFACE_TOKEN=your_hugging_face_token_here\n")
        print("📝 Created .env file. Please add your Hugging Face token.")

# Ensure .env file exists
ensure_env_file()


class RAGDemoSystem:
    """
    Complete RAG system replicating Demo.ipynb functionality
    """
    
    def __init__(
        self,
        model_name: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        model_file: str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        local_dir: str = "./llama-models/",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "chroma_vectorstore",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        max_tokens: int = 150,
        temperature: float = 0.1,
        top_p: float = 0.8,
        repeat_penalty: float = 2.0,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_threads: int = 8,
        retrieval_k: int = 3,
        verbose: bool = True
    ):
        """Initialize RAG Demo System with configuration"""
        self.model_name = model_name
        self.model_file = model_file
        self.local_dir = Path(local_dir)
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.retrieval_k = retrieval_k
        self.verbose = verbose
        
        # Initialize components
        self.model_path: Optional[Path] = None
        self.llm_cpp: Optional[LlamaCpp] = None
        self.vectordb: Optional[Chroma] = None
        self.rag_chain: Optional[RetrievalQA] = None
        self.direct_prompt: Optional[PromptTemplate] = None
        self.rag_prompt: Optional[PromptTemplate] = None
        
        # Setup prompts
        self._setup_prompts()
    
    def _print(self, message: str, emoji: str = "📝", **kwargs):
        """Print with emoji if verbose"""
        if self.verbose:
            print(f"{emoji} {message}", **kwargs)
    
    def _setup_prompts(self):
        """Setup prompt templates"""
        # Direct LLM prompt
        direct_template = """You are a helpful AI assistant. Answer the following question to the best of your knowledge in 1000 characters at maximum.

Question: {question}

Answer:"""
        
        self.direct_prompt = PromptTemplate(
            template=direct_template,
            input_variables=['question']
        )
        
        # RAG prompt
        rag_template = """You are a helpful AI assistant. Answer the question using ONLY the provided context. Give a concise answer in maximum 100 words. Do NOT repeat information.

Context: {context}

Question: {question}

Direct answer:"""
        
        self.rag_prompt = PromptTemplate(
            template=rag_template,
            input_variables=['context', 'question']
        )
        
        self._print("Prompt templates created successfully!")
    
    def setup_huggingface_login(self) -> bool:
        """Setup Hugging Face login"""
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not hf_token or hf_token == 'your_hugging_face_token_here':
            self._print("⚠️  Hugging Face token not found or not set.", "⚠️")
            self._print("📝 Please set your token in the .env file:", "📝")
            self._print("   1. Open the .env file in this directory", "   ")
            self._print("   2. Replace 'your_hugging_face_token_here' with your actual token", "   ")
            self._print("   3. Get your token from: https://huggingface.co/settings/tokens", "   ")
            self._print("   4. Save the .env file and run the script again", "   ")
            
            if input("\nDo you want to enter token manually for this session? (y/n): ").lower() == 'y':
                hf_token = input("Enter your Hugging Face token: ")
            
        if hf_token and hf_token != 'your_hugging_face_token_here':
            try:
                login(token=hf_token)
                self._print("Successfully logged in to Hugging Face Hub", "✅")
                return True
            except Exception as e:
                self._print(f"Failed to login: {e}", "❌")
                return False
        else:
            self._print("Skipping Hugging Face login - some models may not be accessible", "⚠️")
            return False
    
    def download_model(self) -> bool:
        """Download GGUF model from Hugging Face"""
        self._print("Downloading Llama 8B model in GGUF format...", "📥")
        self._print(f"Model: {self.model_name}", "🤖")
        self._print(f"File: {self.model_file}", "📁")
        self._print(f"Saving to: {self.local_dir}", "💾")
        
        # Create directory if it doesn't exist
        self.local_dir.mkdir(exist_ok=True)
        
        # Check if model already exists
        self.model_path = self.local_dir / self.model_file
        if self.model_path.exists():
            self._print(f"Model already exists at: {self.model_path}", "✅")
            return True
        
        try:
            access_token = os.getenv('HUGGINGFACE_TOKEN')
            self._print("Downloading model (this may take a while - ~4.9GB)...", "⬇️")
            
            downloaded_path = hf_hub_download(
                repo_id=self.model_name,
                filename=self.model_file,
                local_dir=str(self.local_dir),
                token=access_token
            )
            
            self.model_path = Path(downloaded_path)
            self._print("Model downloaded successfully!", "✅")
            self._print(f"Model path: {self.model_path}", "📍")
            self._print("Model size: ~4.9GB (4-bit quantized)", "📊")
            return True
            
        except Exception as e:
            self._print(f"Failed to download model: {e}", "❌")
            return False
    
    def load_llm(self) -> bool:
        """Load language model with llama-cpp-python"""
        if not self.model_path or not self.model_path.exists():
            self._print("Model not found. Please download first.", "❌")
            return False
        
        self._print("Loading Llama 8B with llama-cpp-python...", "🤖")
        
        try:
            # Set up callback manager for streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Configure LlamaCpp
            self.llm_cpp = LlamaCpp(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
                callback_manager=callback_manager,
                verbose=False,
                streaming=True,
                stop=["Question:", "Context:", "\n\n", "Note:", "Answer:", 
                      "Human:", "Assistant:", "fig.", "design is"]
            )
            
            self._print("Llama 8B loaded successfully with llama-cpp-python!", "✅")
            return True
            
        except Exception as e:
            self._print(f"Failed to load model: {e}", "❌")
            return False
    
    def load_and_process_pdf(self, pdf_path: str) -> bool:
        """Load and process PDF document"""
        if not os.path.exists(pdf_path):
            self._print(f"PDF file not found: {pdf_path}", "❌")
            return False
        
        self._print("Loading PDF document...", "📄")
        
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Extract text
            all_page_text = [p.page_content for p in pages]
            joined_page_text = " ".join(all_page_text)
            
            self._print(f"Loaded {len(pages)} pages", "✅")
            self._print(f"Total characters: {len(joined_page_text):,}", "📊")
            
            # Split text into chunks
            self._print("Splitting text into chunks...", "✂️")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            splits = text_splitter.split_text(joined_page_text)
            self._print(f"Created {len(splits)} text chunks", "✅")
            self._print(f"Average chunk size: {sum(len(chunk) for chunk in splits) // len(splits)} characters", "📏")
            
            # Create embeddings and vector store
            self._print("Creating embeddings and vector store...", "🔢")
            self._print(f"Using embedding model: {self.embedding_model}", "🤖")
            self._print(f"Vector store location: {self.persist_directory}", "💾")
            
            # Clean up existing vector store
            if os.path.exists(self.persist_directory):
                self._print("Cleaning up existing vector store...", "🧹")
                shutil.rmtree(self.persist_directory)
            
            # Initialize embedding model
            embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
            
            # Create vector store
            self.vectordb = Chroma.from_texts(
                texts=splits,
                embedding=embedding,
                persist_directory=self.persist_directory
            )
            
            # Persist the vector store
            self.vectordb.persist()
            
            self._print(f"Vector store created with {len(splits)} documents", "✅")
            self._print("Ready for similarity search and retrieval!", "🎯")
            return True
            
        except Exception as e:
            self._print(f"Failed to process PDF: {e}", "❌")
            return False
    
    def build_rag_chain(self) -> bool:
        """Build RAG retrieval chain"""
        if not self.llm_cpp:
            self._print("LLM not loaded. Please load model first.", "❌")
            return False
        
        if not self.vectordb:
            self._print("Vector store not created. Please process PDF first.", "❌")
            return False
        
        self._print("Creating RAG retrieval chain with Llama 8B...", "🔗")
        
        try:
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm_cpp,
                chain_type='stuff',
                retriever=self.vectordb.as_retriever(search_kwargs={'k': self.retrieval_k}),
                chain_type_kwargs={'prompt': self.rag_prompt},
                return_source_documents=False
            )
            
            self._print("RAG pipeline created successfully!", "✅")
            self._print("Configuration:", "🎯")
            self._print("  - Model: Llama 3.1 8B Instruct (4-bit quantized)", "  ")
            self._print("  - Backend: llama-cpp-python (optimized)", "  ")
            self._print(f"  - Retrieval: Top {self.retrieval_k} most similar documents", "  ")
            self._print(f"  - Context: {self.n_ctx} tokens", "  ")
            self._print("  - Chain type: Stuff (concatenate all retrieved docs)", "  ")
            self._print("  - Streaming: Enabled", "  ")
            return True
            
        except Exception as e:
            self._print(f"Failed to create RAG chain: {e}", "❌")
            return False
    
    def query_direct_llm(self, question: str) -> str:
        """Query LLM directly without RAG"""
        if not self.llm_cpp:
            raise RuntimeError("LLM not loaded")
        
        formatted_prompt = self.direct_prompt.format(question=question)
        response = self.llm_cpp(formatted_prompt)
        return response.strip()
    
    def query_rag(self, question: str) -> str:
        """Query RAG system"""
        if not self.rag_chain:
            raise RuntimeError("RAG chain not built")
        
        result = self.rag_chain.invoke({"query": question})
        return result['result'] if 'result' in result else str(result)
    
    def compare_responses(self, question: str) -> Dict[str, str]:
        """Compare direct LLM vs RAG responses"""
        self._print("Testing Direct LLM vs RAG Comparison", "🔄")
        self._print("=" * 60)
        self._print(f"Test Question: {question}", "❓")
        self._print("=" * 30)
        
        # Direct LLM response
        self._print("1. DIRECT LLM RESPONSE (No Document Context):", "🤖")
        self._print("-" * 40)
        self._print("Model will use only its training knowledge...", "💭")
        
        try:
            direct_response = self.query_direct_llm(question)
        except Exception as e:
            direct_response = f"Error: {e}"
        
        # RAG response
        self._print("\n2. RAG LLM RESPONSE (With Document Context):", "🧠")
        self._print("-" * 40)
        self._print("Model will use retrieved document context...", "📚")
        self._print("RAG Answer: ", "📝", end="", flush=True)
        
        try:
            rag_response = self.query_rag(question)
        except Exception as e:
            rag_response = f"Error: {e}"
        
        return {
            'question': question,
            'direct_response': direct_response,
            'rag_response': rag_response
        }
    
    def interactive_mode(self):
        """Interactive query mode"""
        self._print("Interactive RAG Mode", "🎮")
        self._print("Type 'quit' to exit, 'compare <question>' to compare responses")
        self._print("-" * 50)
        
        while True:
            try:
                user_input = input("\n❓ Enter your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self._print("Goodbye!", "👋")
                    break
                
                if user_input.lower().startswith('compare '):
                    question = user_input[8:].strip()
                    if question:
                        self.compare_responses(question)
                    else:
                        self._print("Please provide a question after 'compare'", "⚠️")
                    continue
                
                if not user_input:
                    self._print("Please enter a question", "⚠️")
                    continue
                
                # Regular RAG query
                self._print("RAG Answer: ", "📝", end="", flush=True)
                response = self.query_rag(user_input)
                
            except KeyboardInterrupt:
                self._print("\nGoodbye!", "👋")
                break
            except Exception as e:
                self._print(f"Error: {e}", "❌")
    
    def batch_process(self, queries_file: str) -> List[Dict[str, str]]:
        """Process queries from file"""
        if not os.path.exists(queries_file):
            self._print(f"Queries file not found: {queries_file}", "❌")
            return []
        
        results = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        self._print(f"Processing {len(queries)} queries from {queries_file}", "📋")
        
        for i, query in enumerate(queries, 1):
            self._print(f"Query {i}/{len(queries)}: {query}", "🔍")
            try:
                result = self.compare_responses(query)
                results.append(result)
            except Exception as e:
                self._print(f"Error processing query: {e}", "❌")
                results.append({
                    'question': query,
                    'direct_response': f"Error: {e}",
                    'rag_response': f"Error: {e}"
                })
        
        return results
    
    def setup_complete_system(self, pdf_path: str) -> bool:
        """Setup complete RAG system"""
        self._print("Setting up complete RAG system...", "🚀")
        
        # Step 1: Setup Hugging Face login
        if not self.setup_huggingface_login():
            self._print("Warning: Continuing without HF login", "⚠️")
        
        # Step 2: Download model
        if not self.download_model():
            return False
        
        # Step 3: Load LLM
        if not self.load_llm():
            return False
        
        # Step 4: Process PDF and create vectors
        if not self.load_and_process_pdf(pdf_path):
            return False
        
        # Step 5: Build RAG chain
        if not self.build_rag_chain():
            return False
        
        self._print("Complete RAG system setup successfully!", "🎉")
        self._print("Ready for queries!", "✅")
        return True


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="RAG Demo Script - Complete Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_rag_script.py --setup --pdf ./Data/document.pdf
  python demo_rag_script.py --query "What is DRAGON?" --pdf ./Data/document.pdf
  python demo_rag_script.py --interactive --pdf ./Data/document.pdf
  python demo_rag_script.py --compare --query "What is DRAGON?" --pdf ./Data/document.pdf
  python demo_rag_script.py --batch queries.txt --pdf ./Data/document.pdf
        """
    )
    
    # Required arguments
    parser.add_argument("--pdf", type=str, 
                       default="./Data/Dynamic_Resource_Scheduler_for_Distributed_Deep_Learning_Training_in_Kubernetes.pdf",
                       help="Path to PDF document (default: %(default)s)")
    
    # Mode arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--setup", action="store_true", 
                           help="Setup system only (download model, process PDF)")
    mode_group.add_argument("--query", type=str, 
                           help="Single query to process")
    mode_group.add_argument("--interactive", action="store_true", 
                           help="Interactive query mode")
    mode_group.add_argument("--compare", action="store_true", 
                           help="Compare direct LLM vs RAG (requires --query)")
    mode_group.add_argument("--batch", type=str, 
                           help="Batch process queries from file")
    
    # Configuration arguments
    parser.add_argument("--model-name", type=str, 
                       default="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                       help="Hugging Face model name (default: %(default)s)")
    parser.add_argument("--model-file", type=str, 
                       default="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                       help="GGUF model file name (default: %(default)s)")
    parser.add_argument("--local-dir", type=str, default="./llama-models/",
                       help="Local directory for model storage (default: %(default)s)")
    parser.add_argument("--embedding-model", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model name (default: %(default)s)")
    parser.add_argument("--persist-dir", type=str, default="chroma_vectorstore",
                       help="Vector store persistence directory (default: %(default)s)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Text chunk size (default: %(default)s)")
    parser.add_argument("--chunk-overlap", type=int, default=150,
                       help="Text chunk overlap (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=150,
                       help="Maximum tokens to generate (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for text generation (default: %(default)s)")
    parser.add_argument("--top-p", type=float, default=0.8,
                       help="Top-p for text generation (default: %(default)s)")
    parser.add_argument("--repeat-penalty", type=float, default=2.0,
                       help="Repeat penalty (default: %(default)s)")
    parser.add_argument("--retrieval-k", type=int, default=3,
                       help="Number of documents to retrieve (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.compare and not args.query:
        parser.error("--compare requires --query")
    
    # Initialize RAG system
    rag_system = RAGDemoSystem(
        model_name=args.model_name,
        model_file=args.model_file,
        local_dir=args.local_dir,
        embedding_model=args.embedding_model,
        persist_directory=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        retrieval_k=args.retrieval_k,
        verbose=not args.quiet
    )
    
    try:
        # Setup system
        if not rag_system.setup_complete_system(args.pdf):
            print("❌ Failed to setup RAG system")
            sys.exit(1)
        
        # Execute based on mode
        if args.setup:
            print("✅ Setup completed successfully!")
            
        elif args.query:
            if args.compare:
                rag_system.compare_responses(args.query)
            else:
                print("📝 RAG Answer: ", end="", flush=True)
                response = rag_system.query_rag(args.query)
                
        elif args.interactive:
            rag_system.interactive_mode()
            
        elif args.batch:
            results = rag_system.batch_process(args.batch)
            
            # Save results
            output_file = f"batch_results_{Path(args.batch).stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Question: {result['question']}\n")
                    f.write(f"Direct LLM: {result['direct_response']}\n")
                    f.write(f"RAG Response: {result['rag_response']}\n")
                    f.write("-" * 80 + "\n")
            
            print(f"✅ Batch results saved to: {output_file}")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 


#Instructions to run the script:
#1. Install the requirements: pip install -r requirements.txt
#2. export HUGGINGFACE_TOKEN=your_token_here 
#3. Run the script: python demo_rag_script.py --setup --pdf ./Data/Dynamic_Resource_Scheduler_for_Distributed_Deep_Learning_Training_in_Kubernetes.pdf
#4. Run the script: python demo_rag_script.py --query "What is DRAGON?" --pdf ./Data/Dynamic_Resource_Scheduler_for_Distributed_Deep_Learning_Training_in_Kubernetes.pdf
#5. Run the script: python demo_rag_script.py --interactive --pdf ./Data/Dynamic_Resource_Scheduler_for_Distributed_Deep_Learning_Training_in_Kubernetes.pdf
#5. Run the script: python demo_rag_script.py --compare --query "What is DRAGON?" --pdf ./Data/Dynamic_Resource_Scheduler_for_Distributed_Deep_Learning_Training_in_Kubernetes.pdf
#6. Run the script: python demo_rag_script.py --batch queries.txt --pdf ./Data/Dynamic_Resource_Scheduler_for_Distributed_Deep_Learning_Training_in_Kubernetes.pdf 
