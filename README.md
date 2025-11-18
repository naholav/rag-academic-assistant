# RAG Academic Assistant

Retrieval-Augmented Generation system for academic paper Q&A using Qwen3-4B-Instruct and ChromaDB.

## Features

- Multi-PDF support with source attribution
- Dual chat modes: RAG (document-grounded) and Direct Chat (conversation with memory)
- Hybrid retrieval combining semantic search and BM25 keyword matching
- Cross-encoder reranking for precision
- Persistent vector store with ChromaDB
- Streaming responses with real-time generation
- Streamlit interface with adjustable parameters
- Multilingual support (Turkish/English with automatic detection)
- Token-aware chunking with 800-token chunks and 150-token overlap

## Architecture

```
┌─────────────────┐
│  PDF Document   │
│ (PyMuPDF Load)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunking &     │
│  Tokenization   │
│  (800 tokens)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │
│  (all-mpnet-    │
│   base-v2)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChromaDB       │
│  Vector Store   │
│  (Persistent)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hybrid Retrieval│
│  Semantic +     │
│  BM25 + RRF     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reranking      │
│  (Cross-Encoder)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Qwen3-4B LLM   │
│  Generation     │
│  (Streaming)    │
└─────────────────┘
```

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended) or CPU
- HuggingFace account and API token
- 16GB RAM minimum (32GB recommended)
- 10GB disk space for models

## Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd /path/to/rag-academic-assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your HuggingFace token
# Get token from: https://huggingface.co/settings/tokens
nano .env  # or use your preferred editor
```

Your `.env` should look like:
```
HF_TOKEN=hf_your_actual_token_here
```

### 3. Add PDF Documents

Place your PDF files in the project root directory.

Note: PDF files are not included in this repository due to copyright. You need to provide your own research papers.

Example PDFs used during development:
- OpenCodeReasoning.pdf
- StopOverthinking.pdf
- FaithLM.pdf

To configure your PDFs, edit `rag_system.py` and update the `PDF_PATHS` list:
```python
PDF_PATHS = [
    "YourPaper1.pdf",
    "YourPaper2.pdf",
    "YourPaper3.pdf"
]
```

### 4. Run the Application

```bash
streamlit run rag_system.py
```

The application will open automatically at `http://localhost:8501`

## First Run Behavior

On the first run, the system will:

1. **Process PDFs** (10-20 minutes for multiple PDFs)
   - Extract text from all pages across all PDFs
   - Split into 800-token chunks with 150-token overlap
   - Extract metadata (page numbers, PDF names, sections)

2. **Generate Embeddings** (5-10 minutes)
   - Compute BERT embeddings for all chunks
   - Store in ChromaDB with persistence
   - Build BM25 index for keyword search

3. **Download Models** (one-time, ~10GB)
   - Qwen3-4B-Instruct-2507 model (~8GB)
   - Cross-encoder reranker (~100MB)
   - Sentence transformer (~500MB)

**Subsequent runs will be instant** as all data is cached locally.

## Usage

### Chat Modes

**RAG Mode (Default)**
- Document-grounded answers using hybrid search
- Shows source attribution (PDF name and page number)
- No conversation history (independent questions)
- Adjustable retrieval and generation parameters

**Direct Chat Mode**
- General conversation with the model
- 10-message history (5 question-answer pairs)
- No document search
- Maintains conversation context

Toggle between modes using the sidebar switch.

### Tips

- Include specific details (model names, metrics, sections) in questions
- Use technical keywords from papers for better retrieval
- Adjust "Candidates to Retrieve" parameter for broader or narrower search
- Check source panel to verify answer attribution
- Ask in Turkish or English (automatic language detection)

## System Components

### Document Processor
- Library: PyMuPDF (fitz)
- Chunking: 800 tokens with 150-token overlap
- Encoding: tiktoken (cl100k_base)
- Metadata: Page numbers, chunk IDs, token counts, PDF names

### Vector Store
- Database: ChromaDB (persistent mode)
- Embeddings: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
- Index: HNSW with cosine similarity
- Location: ./chroma_db/

### Retrieval System
- Semantic Search: ChromaDB vector similarity
- Keyword Search: BM25 Okapi algorithm
- Fusion: Reciprocal Rank Fusion
- Default weights: Semantic (70%) + BM25 (30%)
- Default candidates: Top 10 retrieved, Top 5 after reranking
- All parameters adjustable in UI

### Reranking
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Purpose: Precision reranking of hybrid results

### Language Model
- Model: Qwen/Qwen3-4B-Instruct-2507
- Parameters: 4 billion
- Precision: FP16 (GPU) / FP32 (CPU)
- Context window: 32K tokens
- Default settings: temperature=0.3, max_tokens=2048, top_p=0.9
- Streaming: Enabled
- Multilingual: Automatic language detection (Turkish/English)

## Configuration

### UI Parameters

RAG Mode settings:
- Candidates to Retrieve (10-100): Chunks to retrieve before reranking
- Final Chunks Used (1-20): Chunks to use after reranking
- Semantic Search Weight (0.0-1.0): Vector similarity weight
- BM25 Keyword Weight (0.0-1.0): Keyword matching weight

Generation settings:
- Temperature (0.1-1.0): Controls randomness (lower = more factual)
- Max Response Tokens (256-4096): Maximum response length
- Top P (0.1-1.0): Nucleus sampling parameter

Mode toggle:
- Enable RAG: Switch between document search and direct chat

### Code Parameters (rag_system.py)

```python
CHUNK_SIZE = 800           # Tokens per chunk (same for all PDFs)
CHUNK_OVERLAP = 150        # Overlap between chunks
TOP_K_RETRIEVAL = 10       # Default candidates for reranking
TOP_K_RERANK = 5          # Default final results returned
SEMANTIC_WEIGHT = 0.7     # Default semantic search weight
BM25_WEIGHT = 0.3         # Default keyword search weight
```

### Adding New PDFs

```python
PDF_PATHS = [
    "OpenCodeReasoning.pdf",
    "StopOverthinking.pdf",
    "FaithLM.pdf",
    "YourNewPaper.pdf"  # Add here
]
```

After adding new PDFs, delete `./chroma_db/` and restart to rebuild the database.

## Troubleshooting

### PDF Not Found
```
Error: PDF file not found: OpenCodeReasoning.pdf
```
**Solution**: Ensure `OpenCodeReasoning.pdf` is in the project root directory.

### HuggingFace Token Error
```
Error: HF_TOKEN not found in .env file
```
**Solution**: Create `.env` file with valid `HF_TOKEN=hf_...`

### CUDA Out of Memory
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce `max_new_tokens` to 1024
- Use CPU mode (automatic fallback)
- Close other GPU applications

### ChromaDB Collection Exists
```
Info: Collection already contains X documents. Skipping ingestion.
```
**This is normal**: The system uses cached embeddings.
**To rebuild**: Delete `./chroma_db/` directory and restart.

### Model Download Fails
```
Error: Connection timeout downloading model
```
**Solutions**:
- Check internet connection
- Verify HF_TOKEN is valid
- Try again (downloads resume automatically)

## Performance Optimization

### GPU Acceleration
- Automatic GPU detection
- Mixed precision (FP16) for 2x speedup
- Device mapping for large models

### Caching Strategy
- Vector store persistence (ChromaDB)
- Model weights cached locally
- Streamlit session state for components

### Memory Management
- Batch embedding computation (32 chunks)
- Lazy model loading
- Efficient token encoding

## File Structure

```
rag-academic-assistant/
├── OpenCodeReasoning.pdf          # Source document
├── rag_system.py                  # Main application
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this)
├── .env.example                  # Template for .env
├── README.md                     # This file
├── chroma_db/                    # Vector database (auto-created)
│   └── [chromadb files]
└── cache/                        # Model cache (auto-created)
    └── [huggingface models]
```

## Technical Specifications

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16GB
- Disk: 15GB free
- GPU: Optional

**Recommended**:
- CPU: 8+ cores
- RAM: 32GB
- Disk: 20GB free
- GPU: NVIDIA with 8GB+ VRAM

### Software Dependencies

- Python 3.9+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU)
- ChromaDB 0.4.22
- Streamlit 1.31.0

See `requirements.txt` for complete list.

## Security Notes

- Never commit `.env` file to version control
- Keep HF_TOKEN private
- Use `.env.example` for sharing configuration templates
- Regularly update dependencies for security patches

## Known Limitations

1. **Limited Language Support**: Currently optimized for Turkish and English only
2. **Context Window**: Qwen3-4B has 32K token limit
3. **GPU Memory**: Large documents may require 8GB+ VRAM
4. **Streaming Latency**: First token may take 2-3 seconds
5. **RAG Mode**: No conversation history - each question is independent
6. **Direct Chat Mode**: Limited to 10-message history (5 Q&A pairs)
7. **Uniform Chunking**: Same chunk size (800 tokens) for all PDFs
8. **Manual PDF Addition**: Requires code edit to add new PDFs

## Future Enhancements

Implemented:
- Multi-document support
- Multilingual support (Turkish/English)
- Dual chat modes (RAG + Direct Chat)

Planned:
- Web-based PDF upload interface
- Export chat history
- Custom prompt templates per PDF
- Per-PDF chunk size configuration
- Additional language support
- RAG mode with selective conversation history

## Troubleshooting Commands

```bash
# Check Python version
python --version

# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check installed packages
pip list | grep -E "torch|transformers|chromadb|streamlit"

# Test HuggingFace token
python -c "from huggingface_hub import HfApi; HfApi().whoami()"

# Clear ChromaDB
rm -rf chroma_db/

# Clear model cache
rm -rf cache/
```

## Support

For issues and questions:
- Check troubleshooting section above
- Review error messages in Streamlit interface
- Verify all prerequisites are installed
- Ensure `.env` is configured correctly

## License

This project is provided as-is for educational and research purposes.

## Credits

- LLM: Qwen Team (Alibaba Cloud)
- Embeddings: Sentence Transformers
- Vector DB: ChromaDB
- UI Framework: Streamlit
- PDF Processing: PyMuPDF
