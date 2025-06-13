from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
import os

class RAGPipeline:
    def __init__(
        self,
        model_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "chroma_storage",
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        top_k: int = 3
    ):
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self._load_model()
        self._setup_pipeline()

    def _load_model(self):
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        print("Model loaded ✅")

    def _setup_pipeline(self):
        print("Setting up text generation pipeline...")
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=100,
            repetition_penalty=1.1,
            model_kwargs={"temperature": 0.01}
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def load_and_embed(self, pdf_path: str):
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = " ".join(p.page_content for p in pages)

        print("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)

        print(f"{len(chunks)} chunks created.")

        print("Creating embeddings...")
        embedding_fn = HuggingFaceEmbeddings(model_name=self.embedding_model)

        self.vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_fn,
            persist_directory=self.persist_directory
        )
        self.vectordb.persist()

        print("Embeddings stored and persisted.")

    def build_rag_chain(self):
        print("Creating RAG chain...")

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of information to answer the user's question. Explain the answer clearly.
If you don't know the answer, just say you don't know. Don't make up facts.

Context: {context}
Question: {question}

Helpful answer in 1000 characters or less:
"""
        )

        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectordb.as_retriever(search_kwargs={"k": self.top_k}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, query: str) -> str:
        if not hasattr(self, "rag_chain"):
            raise RuntimeError("You must call build_rag_chain() before asking questions.")
        print(f"Asking RAG: {query}")
        response = self.rag_chain.invoke({"query": query})
        return response["result"]

    def ask_base_model(self, query: str) -> str:
        prompt = f"""Use the following question to provide a helpful answer.
If you don't know the answer, just say that you don't know.

Question: {query}

Helpful answer:
"""
        return self.pipe(prompt)[0]['generated_text'].split("Helpful answer:")[-1].strip()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a RAG question-answering pipeline.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file to load.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the RAG model.")
    parser.add_argument("--model-path", type=str, default="./meta-llama/Llama-3.2-1B", help="Path to the local model directory.")
    parser.add_argument("--use-base", action="store_true", help="Use base model without RAG (no retrieval).")

    args = parser.parse_args()

    rag = RAGPipeline(model_path=args.model_path)
    rag.load_and_embed(args.pdf)
    rag.build_rag_chain()

    if args.use_base:
        answer = rag.ask_base_model(args.question)
    else:
        answer = rag.ask(args.question)

    print("\n✅ Answer:")
    print(answer)

