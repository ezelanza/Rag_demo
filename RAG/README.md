# RAG (Retrieval Augmented Generation) Demo

This repository contains a complete implementation of a RAG system using LangChain, Hugging Face models, and Chroma vector database. The system demonstrates how to build a context-aware question-answering system that can provide accurate responses based on document content.

## 🎯 What is RAG?

RAG is a technique that:
1. **Retrieves** relevant documents from a knowledge base
2. **Augments** the input prompt with this retrieved context  
3. **Generates** a response using a language model

This approach helps overcome the limitations of pure language models by providing them with specific, relevant information to work with.

## 📁 Project Structure

```
RAG/
├── Demo.ipynb                 # Interactive Jupyter notebook demonstration
├── demo_rag_script.py        # Complete command-line RAG implementation
├── rag_script.py             # Original RAG script (transformers-based)
├── sample_queries.txt        # Sample queries for batch processing
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Data/                     # Document storage
│   └── Dynamic_Resource_Scheduler_for_Distributed_Deep_Learning_Training_in_Kubernetes.pdf
├── chroma_vectorstore/       # Vector database storage
└── llama-models/            # Downloaded language models
```

## 🔐 Environment Setup

This project uses a `.env` file to securely store your Hugging Face token:

1. **The .env file will be automatically created** when you first run the script
2. **Edit the .env file** and replace `your_hugging_face_token_here` with your actual token
3. **Get your token** from: https://huggingface.co/settings/tokens
4. **The .env file is automatically ignored by git** (never committed to version control)

Example `.env` file content:
```
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```

## 🚀 Quick Start

### Option 1: Interactive Notebook (Recommended for Learning)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Hugging Face token:**
   - Open the `.env` file in the RAG directory
   - Replace `your_hugging_face_token_here` with your actual token
   - Get your token from: https://huggingface.co/settings/tokens

3. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook Demo.ipynb
   ```

4. **Follow the step-by-step cells** to understand the RAG implementation.

### Option 2: Command-Line Script (Recommended for Production)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Hugging Face token:**
   - Open the `.env` file in the RAG directory
   - Replace `your_hugging_face_token_here` with your actual token
   - Get your token from: https://huggingface.co/settings/tokens

3. **Initial setup (downloads model, processes PDF):**
   ```bash
   python demo_rag_script.py --setup
   ```

4. **Ask a single question:**
   ```bash
   python demo_rag_script.py --query "What is DRAGON?"
   ```

5. **Compare direct LLM vs RAG responses:**
   ```bash
   python demo_rag_script.py --compare --query "What is DRAGON?"
   ```

6. **Interactive mode:**
   ```bash
   python demo_rag_script.py --interactive
   ```

7. **Batch processing:**
   ```bash
   python demo_rag_script.py --batch sample_queries.txt
   ```

## 🛠️ Implementation Details

### Key Components

1. **Language Model**: Llama 3.1 8B Instruct (4-bit quantized GGUF format)
   - Optimized for CPU inference with llama-cpp-python
   - ~4.9GB memory footprint
   - Streaming response support

2. **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
   - Lightweight and fast
   - Good performance for document similarity

3. **Vector Store**: Chroma
   - Persistent storage for document embeddings
   - Efficient similarity search

4. **Document Processing**: 
   - PDF loading with PyPDFLoader
   - Text chunking with RecursiveCharacterTextSplitter
   - Configurable chunk size and overlap

### System Architecture

```
PDF Document → Text Chunks → Embeddings → Vector Store
                                            ↓
User Query → Embedding → Similarity Search → Retrieved Context
                                            ↓
Retrieved Context + Query → LLM → Final Response
```

## 📊 Performance Comparison

The system demonstrates clear advantages of RAG over base language models:

| Aspect | Direct LLM | RAG System |
|--------|------------|------------|
| **Knowledge Source** | Training data only | Document-specific |
| **Accuracy** | Generic responses | Context-aware responses |
| **Relevance** | May hallucinate | Grounded in documents |
| **Updatable** | Requires retraining | Just add new documents |

### Example Comparison

**Question**: "What is DRAGON?"

**Direct LLM**: Generic information about dragons (mythical creatures)

**RAG System**: "DRAGON is a resource scheduler that schedules distributed jobs using gang scheduling and autoscaling." *(from the actual document)*

## ⚙️ Configuration Options

### Command-Line Arguments

```bash
python demo_rag_script.py --help
```

Key configuration options:
- `--model-name`: Hugging Face model repository
- `--model-file`: Specific GGUF model file
- `--chunk-size`: Text chunk size (default: 1000)
- `--chunk-overlap`: Chunk overlap (default: 150)
- `--max-tokens`: Maximum tokens to generate (default: 150)
- `--temperature`: Generation temperature (default: 0.1)
- `--retrieval-k`: Number of documents to retrieve (default: 3)

### Customization

You can easily customize the system for your own documents:

1. **Replace the PDF**: Update the `--pdf` argument
2. **Adjust chunk size**: Modify `--chunk-size` based on your document structure
3. **Change the model**: Use different `--model-name` and `--model-file`
4. **Tune retrieval**: Adjust `--retrieval-k` for more/fewer context documents

## 🔧 Advanced Usage

### Using Different Models

```bash
# Use a different model
python demo_rag_script.py --model-name "microsoft/DialoGPT-large" \
                         --model-file "DialoGPT-large-Q4_K_M.gguf" \
                         --query "Your question here"
```

### Custom PDF Processing

```bash
# Process a different PDF with custom settings
python demo_rag_script.py --pdf "/path/to/your/document.pdf" \
                         --chunk-size 1500 \
                         --chunk-overlap 200 \
                         --retrieval-k 5 \
                         --interactive
```

### Batch Processing with Custom Queries

Create a text file with your questions (one per line) and run:

```bash
python demo_rag_script.py --batch your_queries.txt
```

## 🎓 Learning Path

1. **Start with the Notebook**: `Demo.ipynb` provides step-by-step explanation
2. **Experiment with the Script**: Use `demo_rag_script.py` for different queries
3. **Compare Responses**: Use `--compare` to see RAG vs Direct LLM differences
4. **Try Your Own Documents**: Replace the PDF with your own content
5. **Tune Parameters**: Experiment with different chunk sizes and retrieval settings

## 🔍 Troubleshooting

### Common Issues

1. **Model Download Fails**:
   - Check your internet connection
   - Verify Hugging Face token is set correctly
   - Ensure sufficient disk space (~5GB)

2. **Memory Issues**:
   - Reduce `--max-tokens`
   - Use smaller chunk sizes
   - Reduce `--retrieval-k`

3. **Slow Performance**:
   - Increase `--n-threads` for more CPU cores
   - Use GPU-enabled models if available
   - Reduce context window size

4. **Poor Responses**:
   - Increase `--retrieval-k` for more context
   - Adjust `--chunk-size` for better text segmentation
   - Modify prompts for better instruction following

### System Requirements

- **Minimum**: 8GB RAM, 10GB disk space
- **Recommended**: 16GB RAM, 20GB disk space, 8+ CPU cores
- **GPU**: Optional but recommended for larger models

## 📚 Dependencies

Main dependencies (see `requirements.txt` for full list):
- `langchain` - RAG framework
- `llama-cpp-python` - Optimized LLM inference
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `transformers` - Hugging Face models
- `pypdf` - PDF processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both notebook and script
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for providing pre-trained models
- LangChain for the RAG framework
- Chroma for the vector database
- The open-source community for the underlying technologies

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the example usage
3. Open an issue on GitHub
4. Review the Jupyter notebook for detailed explanations

---

**Happy RAG-ing! 🚀** 