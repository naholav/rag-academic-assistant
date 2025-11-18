"""
RAG Academic Assistant - Production-Ready System
Powered by Qwen3-4B-Instruct-2507 and ChromaDB

Author: naholav
Created: 2025-11-18
Description: Advanced RAG system for academic paper Q&A with hybrid retrieval,
             cross-encoder reranking, and streaming LLM responses.
"""

import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import torch
from dotenv import load_dotenv
import fitz  # PyMuPDF
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from rank_bm25 import BM25Okapi
import tiktoken
from threading import Thread

# Load environment variables
load_dotenv()

# Configuration
PDF_PATHS = [
    "OpenCodeReasoning.pdf",
    "StopOverthinking.pdf",
    "FaithLM.pdf"
]
CHROMA_DIR = "./chroma_db"
CACHE_DIR = "./cache"
COLLECTION_NAME = "academic_papers"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_name():
    """Get human-readable device name for UI display"""
    device = get_device()
    if device == "cuda":
        return "GPU (CUDA)"
    elif device == "mps":
        return "GPU (Apple Silicon)"
    else:
        return "CPU"

# ============================================================================
# DOCUMENT PROCESSING MODULE
# ============================================================================

class DocumentProcessor:
    """Process PDF documents with chunking and metadata extraction"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def load_pdf(self, pdf_path: str) -> List[Dict]:
        """Load and extract text from PDF with metadata"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Extract PDF name without extension
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')

        doc = fitz.open(pdf_path)
        chunks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Chunk the page text
            page_chunks = self._chunk_text(text, page_num + 1, pdf_name)
            chunks.extend(page_chunks)

        doc.close()
        return chunks

    def _chunk_text(self, text: str, page_num: int, pdf_name: str) -> List[Dict]:
        """Split text into overlapping chunks with token-based sizing"""
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "page_number": page_num,
                "chunk_id": f"{pdf_name}_p{page_num}_c{chunk_id}",
                "token_count": len(chunk_tokens),
                "pdf_name": pdf_name
            })

            chunk_id += 1
            start += self.chunk_size - self.overlap

        return chunks

# ============================================================================
# EMBEDDING AND VECTOR STORE MODULE
# ============================================================================

class VectorStore:
    """ChromaDB-based vector storage with persistence"""

    def __init__(self, persist_directory: str = CHROMA_DIR):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
            )
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

    def add_documents(self, chunks: List[Dict]) -> None:
        """Add documents to vector store with embeddings"""
        if self.collection.count() > 0:
            st.info(f"Collection already contains {self.collection.count()} documents. Skipping ingestion.")
            return

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "token_count": chunk["token_count"],
            "pdf_name": chunk["pdf_name"]
        } for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]

        # Generate embeddings in batches
        batch_size = 32
        progress_bar = st.progress(0)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()

            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            progress = (i + batch_size) / len(texts)
            progress_bar.progress(min(progress, 1.0))

        progress_bar.empty()
        st.success(f"Added {len(texts)} chunks to vector store")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Semantic search using ChromaDB"""
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })

        return chunks

    def get_all_documents(self) -> List[str]:
        """Retrieve all documents for BM25 indexing"""
        results = self.collection.get(include=["documents"])
        return results["documents"]

# ============================================================================
# HYBRID RETRIEVAL MODULE
# ============================================================================

class HybridRetriever:
    """Hybrid retrieval combining semantic search and BM25"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.bm25 = None
        self.all_chunks = []
        self.semantic_weight = SEMANTIC_WEIGHT
        self.bm25_weight = BM25_WEIGHT

    def initialize_bm25(self):
        """Initialize BM25 index"""
        if self.bm25 is not None:
            return

        documents = self.vector_store.get_all_documents()
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.all_chunks = documents

    def retrieve(self, query: str, top_k: int = TOP_K_RERANK, top_k_retrieval: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Hybrid retrieval with reranking"""
        # Semantic search
        semantic_results = self.vector_store.search(query, top_k_retrieval)

        # BM25 search
        self.initialize_bm25()
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_retrieval]
        bm25_results = [
            {"text": self.all_chunks[idx], "score": bm25_scores[idx]}
            for idx in top_bm25_indices
        ]

        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results, bm25_results
        )

        # Rerank with cross-encoder
        reranked_results = self._rerank(query, fused_results, top_k)

        return reranked_results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict]
    ) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion"""
        k = 60  # RRF constant
        scores = {}
        documents = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results):
            text = result["text"]
            score = self.semantic_weight / (k + rank + 1)
            scores[text] = scores.get(text, 0) + score
            documents[text] = result

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            text = result["text"]
            score = self.bm25_weight / (k + rank + 1)
            scores[text] = scores.get(text, 0) + score
            if text not in documents:
                documents[text] = result

        # Sort by fused score
        sorted_texts = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused_results = []
        for text in sorted_texts[:TOP_K_RETRIEVAL]:
            result = documents[text].copy()
            result["fusion_score"] = scores[text]
            fused_results.append(result)

        return fused_results

    def _rerank(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """Rerank results using cross-encoder"""
        if not results:
            return []

        pairs = [[query, result["text"]] for result in results]
        rerank_scores = self.reranker.predict(pairs)

        for i, result in enumerate(results):
            result["rerank_score"] = float(rerank_scores[i])

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

# ============================================================================
# QWEN MODEL MODULE
# ============================================================================

class QwenGenerator:
    """Qwen3-4B-Instruct model for response generation"""

    def __init__(self):
        self.model_name = "Qwen/Qwen3-4B-Instruct-2507"
        self.device = get_device()
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load Qwen model with authentication"""
        if self.model is not None:
            return

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in .env file")

        with st.spinner(f"Loading {self.model_name} (downloading if not cached, ~8GB)..."):
            st.info(f"📥 Checking for model cache... If not found, will download from HuggingFace")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=True
            )

            st.info(f"✓ Tokenizer loaded. Loading model weights...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            st.success(f"Model loaded on {self.device}")

    def generate(
        self,
        query: str,
        context: List[Dict],
        stream: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 0.9
    ):
        """Generate response with streaming support (no chat history in RAG mode)"""
        self.load_model()

        # Format context
        context_text = "\n\n".join([
            f"[Source {i+1} - {c.get('metadata', {}).get('pdf_name', 'Unknown')} - Page {c.get('metadata', {}).get('page_number', 'N/A')}]\n{c['text']}"
            for i, c in enumerate(context)
        ])

        # Build messages list - no chat history in RAG mode
        # Create prompt with strong language instruction
        prompt = f"""You are an academic assistant helping to answer questions about research papers. Use the provided context to answer the question accurately and concisely.

CRITICAL INSTRUCTION: You MUST answer in the SAME LANGUAGE as the question below. If the question is in Turkish, answer ONLY in Turkish. If the question is in English, answer ONLY in English. This is absolutely mandatory.

Context:
{context_text}

Question: {query}

Answer: Provide a detailed answer based on the context above. REMEMBER: Answer in the SAME LANGUAGE as the question! Include specific references to page numbers when relevant. If the context doesn't contain enough information, acknowledge the limitation."""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        if stream:
            # Streaming generation
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=1.1
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            return streamer
        else:
            # Non-streaming generation
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=1.1
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

    def generate_direct(
        self,
        query: str,
        chat_history: List[Dict] = None,
        stream: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 0.9
    ):
        """Generate response without RAG context but with chat history"""
        self.load_model()

        # Build messages list with chat history
        messages = []

        # Add chat history (last 10 messages = 5 Q&A pairs)
        if chat_history:
            history_to_use = chat_history[-10:]  # Last 10 messages
            for msg in history_to_use:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Create current prompt with strong language instruction
        current_prompt = f"""You are a helpful AI assistant. Answer the following question directly and concisely.

CRITICAL INSTRUCTION: You MUST answer in the SAME LANGUAGE as the question below. If the question is in Turkish, answer ONLY in Turkish. If the question is in English, answer ONLY in English. This is absolutely mandatory.

Question: {query}

Answer (in the SAME language as the question):"""

        messages.append({"role": "user", "content": current_prompt})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        if stream:
            # Streaming generation
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=1.1
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            return streamer
        else:
            # Non-streaming generation
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=1.1
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

def initialize_system():
    """Initialize RAG system components"""
    if st.session_state.initialized:
        return

    try:
        # Check all PDFs exist
        missing_pdfs = [pdf for pdf in PDF_PATHS if not os.path.exists(pdf)]
        if missing_pdfs:
            st.error(f"PDF files not found: {', '.join(missing_pdfs)}")
            st.stop()

        # Initialize vector store
        st.session_state.vector_store = VectorStore(CHROMA_DIR)

        # Process PDFs if needed
        if st.session_state.vector_store.collection.count() == 0:
            st.info(f"Processing {len(PDF_PATHS)} PDFs for the first time (this may take 10-20 minutes)...")
            processor = DocumentProcessor()
            all_chunks = []

            for pdf_path in PDF_PATHS:
                st.write(f"📄 Processing {pdf_path}...")
                chunks = processor.load_pdf(pdf_path)
                all_chunks.extend(chunks)
                st.success(f"✓ {pdf_path}: {len(chunks)} chunks")

            st.session_state.vector_store.add_documents(all_chunks)
        else:
            st.success(f"Loaded {st.session_state.vector_store.collection.count()} chunks from database")

        # Initialize retriever
        st.session_state.retriever = HybridRetriever(st.session_state.vector_store)

        # Initialize generator
        st.session_state.generator = QwenGenerator()

        st.session_state.initialized = True

    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.stop()

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="RAG Academic Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 RAG Academic Assistant")
    st.caption("Powered by Qwen3-4B-Instruct and ChromaDB")

    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Info")

        if st.button("Initialize System", type="primary"):
            st.session_state.initialized = False
            initialize_system()

        if st.session_state.initialized:
            st.success("✅ System Ready")
            st.metric("Documents", st.session_state.vector_store.collection.count())
            st.metric("Model", "Qwen3-4B-Instruct-2507")
            st.metric("Device", get_device_name())

        st.divider()

        st.header("📄 Documents")
        for pdf in PDF_PATHS:
            st.info(f"📝 {pdf}")

        st.divider()

        st.header("💬 Chat Mode")
        use_rag = st.toggle(
            "Enable RAG (Document Search)",
            value=True,
            help="When enabled, searches PDFs for answers. When disabled, chat directly with the model."
        )

        if use_rag:
            st.success("🔍 RAG Mode: Searching documents")
        else:
            st.info("💭 Direct Chat: No document search")

        st.divider()

        st.header("🔧 Settings")

        # Basic Settings
        if use_rag:
            st.subheader("Retrieval")
            top_k_retrieval = st.slider(
                "Candidates to Retrieve",
                min_value=10,
                max_value=100,
                value=TOP_K_RETRIEVAL,
                help="Number of chunks to retrieve before reranking"
            )

            top_k = st.slider(
                "Final Chunks Used",
                min_value=1,
                max_value=20,
                value=TOP_K_RERANK,
                help="Number of chunks after reranking to use for generation"
            )
        else:
            top_k_retrieval = TOP_K_RETRIEVAL
            top_k = TOP_K_RERANK

        # Advanced Settings
        with st.expander("🎛️ Advanced Settings"):
            if use_rag:
                st.write("**Hybrid Retrieval Weights**")
                semantic_weight = st.slider(
                    "Semantic Search Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=SEMANTIC_WEIGHT,
                    step=0.1,
                    help="Weight for semantic similarity search"
                )

                bm25_weight = st.slider(
                    "BM25 Keyword Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=BM25_WEIGHT,
                    step=0.1,
                    help="Weight for keyword-based search"
                )

                st.divider()
            else:
                semantic_weight = SEMANTIC_WEIGHT
                bm25_weight = BM25_WEIGHT

            st.write("**Generation Settings**")
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Lower = more focused, Higher = more creative"
            )

            max_tokens = st.slider(
                "Max Response Tokens",
                min_value=256,
                max_value=4096,
                value=2048,
                step=256,
                help="Maximum length of generated response"
            )

            top_p = st.slider(
                "Top P (Nucleus Sampling)",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Diversity of token selection"
            )

            st.divider()

            st.write("**Document Processing (Info)**")
            st.info(f"📄 Chunk Size: {CHUNK_SIZE} tokens")
            st.info(f"↔️ Chunk Overlap: {CHUNK_OVERLAP} tokens")

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Initialize system if not done
    if not st.session_state.initialized:
        initialize_system()

    # Main chat interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if query := st.chat_input("Ask a question..." if not use_rag else "Ask a question about the documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            if use_rag:
                # RAG MODE: Retrieve context
                with st.spinner("Retrieving relevant context..."):
                    # Update retriever weights if changed
                    st.session_state.retriever.semantic_weight = semantic_weight
                    st.session_state.retriever.bm25_weight = bm25_weight
                    context = st.session_state.retriever.retrieve(
                        query,
                        top_k=top_k,
                        top_k_retrieval=top_k_retrieval
                    )

                # Generate response with context (no chat history in RAG mode)
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    streamer = st.session_state.generator.generate(
                        query,
                        context,
                        stream=True,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )

                    for text in streamer:
                        full_response += text
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                # Store context for display
                st.session_state.last_context = context
            else:
                # DIRECT CHAT MODE: No retrieval
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    # Get chat history before the current message (exclude the just-added user message)
                    chat_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else None

                    streamer = st.session_state.generator.generate_direct(
                        query,
                        chat_history=chat_history,
                        stream=True,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )

                    for text in streamer:
                        full_response += text
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                # Clear context display
                if hasattr(st.session_state, 'last_context'):
                    del st.session_state.last_context

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

    with col2:
        st.subheader("📖 Retrieved Sources")

        if hasattr(st.session_state, "last_context"):
            for chunk in st.session_state.last_context:
                pdf_name = chunk.get('metadata', {}).get('pdf_name', 'Unknown')
                page_num = chunk.get('metadata', {}).get('page_number', 'N/A')
                with st.expander(f"📄 {pdf_name} - Page {page_num}"):
                    st.caption(f"Confidence: {chunk.get('rerank_score', 0):.3f}")
                    st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
        else:
            st.info("Ask a question to see retrieved sources")

if __name__ == "__main__":
    main()
