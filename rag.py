import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
try:
    from groq import Groq
except ImportError:
    Groq = None
    print("⚠️  Warning: 'groq' package not installed. Install with: pip install groq")
except Exception as e:
    Groq = None
    print(f"⚠️  Warning: Failed to import groq: {e}")

DOCS_FOLDER = "docs"


#DATA PARSING FUNCTIONS:
def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t + "\n"
    return text

def load_document(path):
    if path.lower().endswith(".txt"):
        return load_txt(path)
    elif path.lower().endswith(".pdf"):
        return load_pdf(path)
    return None




#CHUNKING FUNCTIONS:(***)
def chunk_text(text, size, strategy="fixed"):
    """
    Chunking strategies for experimental comparison
    
    Improved to maintain sentence boundaries and context
    """
    if strategy == "fixed":
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), size):
            chunk = " ".join(words[i:i+size])
            # Clean up excessive whitespace
            chunk = " ".join(chunk.split())
            chunks.append(chunk)
        
        return chunks
    
    elif strategy == "sentence":
        # Sentence-based chunking (more coherent)
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback if NLTK not available
            sentences = text.split('. ')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_size + sentence_words > size and current_chunk:
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_words
            else:
                current_chunk.append(sentence)
                current_size += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    # Default to fixed chunking
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def load_and_chunk_folder(folder, chunk_size, strategy="fixed"):
    chunks, metas = [], []
    
    for filename in os.listdir(folder):
        fp = os.path.join(folder, filename)
        text = load_document(fp)
        
        if not text:
            continue
        
        file_chunks = chunk_text(text, chunk_size, strategy)
        
        for i, c in enumerate(file_chunks):
            chunks.append(c)
            metas.append({"source": filename, "chunk": i})
    
    return chunks, metas
    

#EMBEDDING MODELS AND EMBEDDING:
EXPERIMENTAL_PIPELINES = {
    "miniLM_fixed100": {
        "model": "all-MiniLM-L6-v2",
        "chunk_size": 100,
        "chunk_strategy": "fixed",
        "description": "Fast, small embeddings with small fixed chunks"
    },
    "miniLM_fixed200": {
        "model": "all-MiniLM-L6-v2",
        "chunk_size": 200,
        "chunk_strategy": "fixed",
        "description": "Fast embeddings with larger context windows"
    },
    "E5small_fixed150": {
        "model": "intfloat/e5-small",
        "chunk_size": 150,
        "chunk_strategy": "fixed",
        "description": "E5 embeddings optimized for semantic search"
    },
    "BGE_fixed150": {
        "model": "BAAI/bge-base-en-v1.5",
        "chunk_size": 150,
        "chunk_strategy": "fixed",
        "description": "BGE embeddings with balanced performance"
    },
    "BGE_fixed250": {
        "model": "BAAI/bge-base-en-v1.5",
        "chunk_size": 250,
        "chunk_strategy": "fixed",
        "description": "BGE embeddings with larger context (better for complex queries)"
    },
}

EMBEDDERS = {}
for name, cfg in EXPERIMENTAL_PIPELINES.items():
    print(f"[EXPERIMENT INIT] Loading embedding model: {cfg['model']}")
    EMBEDDERS[name] = SentenceTransformer(cfg["model"])


# ============================================
# VECTOR DATABASE SETUP
# ============================================

DBS = {}
for name in EXPERIMENTAL_PIPELINES.keys():
    db_path = f"./chroma_db_{name}"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=f"rag_experiment_{name}")
    DBS[name] = collection


# ============================================
# INDEXING PIPELINE
# ============================================

def index_all_pipelines():
    """
    Index documents across all experimental configurations
    """
    for name, cfg in EXPERIMENTAL_PIPELINES.items():
        print(f"\n[EXPERIMENT] Indexing Pipeline: {name}")
        print(f"  Model: {cfg['model']}")
        print(f"  Chunk Size: {cfg['chunk_size']}")
        print(f"  Strategy: {cfg['chunk_strategy']}")
        
        chunks, metas = load_and_chunk_folder(
            DOCS_FOLDER, 
            cfg["chunk_size"],
            cfg["chunk_strategy"]
        )
        
        if len(chunks) == 0:
            print("  ⚠️  No documents found")
            continue
        
        embeddings = EMBEDDERS[name].encode(chunks, show_progress_bar=True)
        embeddings = [e.tolist() for e in embeddings]
        
        ids = [f"{name}_{i}" for i in range(len(chunks))]
        
        try:
            DBS[name].add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metas,
            )
            print(f"  ✅ Indexed {len(chunks)} chunks")
        except Exception:
            print(f"  ℹ️  Already indexed")

# ============================================
# RETRIEVAL SYSTEM
# ============================================
TOP_K_DEFAULT = 5

def retrieve(pipeline_name, query, top_k=TOP_K_DEFAULT):
    """
    Retrieve relevant chunks using specified pipeline
    """
    q_emb = EMBEDDERS[pipeline_name].encode([query])[0].tolist()
    
    result = DBS[pipeline_name].query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )
    
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    
    return docs, metas


# ============================================
# GENERATION SYSTEM
# ============================================
LLAMA_MODELS = {
    "llama-3.1-70b-versatile": "LLaMA 3.1 70B Versatile (Recommended for complex tasks)",
    "llama-3.1-8b-instant": "LLaMA 3.1 8B Instant (Fast responses)",
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B Versatile (Latest, best performance)",
    "llama-3.3-8b-instant": "LLaMA 3.3 8B Instant (Latest, fast)",
    "mixtral-8x7b-32768": "Mixtral 8x7B (32768 context, multilingual)",
    "gemma2-9b-it": "Gemma 2 9B (Google's model)",
    # Legacy models (may be decommissioned)
    "llama3-70b-8192": "LLaMA 3 70B (DEPRECATED - use llama-3.1-70b-versatile)",
    "llama3-8b-8192": "LLaMA 3 8B (DEPRECATED - use llama-3.1-8b-instant)",
}
DEFAULT_LLAMA_MODEL = "llama-3.1-70b-versatile" 

# Initialize LLM client (will be re-initialized dynamically if API key is set later)
groq_client = None

def get_groq_client():
    """
    Get or initialize Groq client dynamically.
    Checks for API key at runtime, allowing it to be set after module import.
    This enables LLaMA models to work even when API key is set via Streamlit UI.
    
    Returns:
        Groq client if successful, None otherwise
    """
    global groq_client
    
    # Check if Groq library is available
    if Groq is None:
        return None
    
    # Check for API key in environment (may have been set after import)
    api_key = os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        return None
    
    # Validate API key format (should start with 'gsk_')
    if not api_key.startswith('gsk_'):
        return None
    
    # Always create a fresh client to handle API key updates
    # This ensures that if API key is set after module import, it will work
    try:
        groq_client = Groq(api_key=api_key)
        return groq_client
    except Exception as e:
        # If initialization fails, don't return stale client
        groq_client = None
        # Return None - error will be handled by caller with better context
        return None



def generate_answer(query, retrieved_chunks, model_name=None):
    """
    Generate answer using LLaMA models via Groq with retrieved context
    Measures information flow: retrieval → generation
    
    Args:
        query: User query
        retrieved_chunks: List of retrieved document chunks
        model_name: Optional LLaMA model name (defaults to DEFAULT_LLAMA_MODEL)
    """
    # Clean and structure the context
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        # Remove excessive whitespace and normalize
        cleaned_chunk = " ".join(chunk.split())
        context_parts.append(f"[Source {i}]\n{cleaned_chunk}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are an AI research assistant. Answer the question using ONLY the provided context from academic sources.

Context from Retrieved Documents:
{context}

Question: {query}

Instructions:
1. Provide a comprehensive, well-structured answer based ONLY on the context above
2. If multiple sources discuss the topic, synthesize the information
3. Use proper paragraph structure and clear explanations
4. If the context contains incomplete information, work with what's available
5. If the context doesn't contain relevant information, explicitly state this
6. Do not add external knowledge or make assumptions beyond the context
7. Cite source numbers when referencing specific information (e.g., "According to Source 1...")

Provide a complete, professional answer:"""
    
    # Try to get Groq client (will check for API key dynamically)
    client = get_groq_client()
    
    if client is None:
        # Provide detailed diagnostic information
        api_key = os.getenv("GROQ_API_KEY", "")
        
        if Groq is None:
            error_msg = (
                "Groq library not installed. Install with: pip install groq\n"
                "Then set your GROQ_API_KEY environment variable or use the Streamlit UI."
            )
        elif not api_key:
            error_msg = (
                "GROQ_API_KEY not set. Please:\n"
                "1. Get a free API key from https://console.groq.com/\n"
                "2. Set it via: export GROQ_API_KEY='your-key-here'\n"
                "3. Or enter it in the Streamlit UI sidebar"
            )
        elif not api_key.startswith('gsk_'):
            error_msg = (
                f"Invalid API key format. Groq API keys should start with 'gsk_'.\n"
                f"Your key appears to start with '{api_key[:7] if len(api_key) >= 7 else '...'}...'\n"
                f"Please verify your API key at https://console.groq.com/"
            )
        else:
            # API key looks valid but client initialization failed
            error_msg = (
                "Failed to initialize Groq client with provided API key.\n"
                "Possible issues:\n"
                "1. API key may be invalid or expired\n"
                "2. Network connectivity issues\n"
                "3. Groq service may be temporarily unavailable\n\n"
                "Please verify your API key at https://console.groq.com/"
            )
        
        return generate_fallback_answer(query, retrieved_chunks, context, error=error_msg)
    
    # Use specified model or default
    model = model_name or DEFAULT_LLAMA_MODEL
    if model not in LLAMA_MODELS:
        model = DEFAULT_LLAMA_MODEL  # Fallback to default if invalid model
    
    # Try the requested model, with automatic fallback to alternatives
    models_to_try = [model]
    
    # If model is deprecated, add current alternatives first
    deprecated_models = ["llama3-70b-8192", "llama3-8b-8192"]
    if model in deprecated_models:
        # Try current model names first
        models_to_try = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant", 
            "llama-3.3-70b-versatile",
            "llama-3.3-8b-instant"
        ]
    
    # Add fallback models if primary fails (prioritize versatile models)
    if "llama-3.1-70b-versatile" not in models_to_try:
        models_to_try.append("llama-3.1-70b-versatile")
    if "llama-3.1-8b-instant" not in models_to_try:
        models_to_try.append("llama-3.1-8b-instant")
    
    last_error = None
    for attempt_model in models_to_try:
        try:
            # Use LLaMA models via Groq
            response = client.chat.completions.create(
                model=attempt_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            # If we used a fallback model, note it in the response
            if attempt_model != model:
                return f"[Note: Using {attempt_model} as {model} is unavailable]\n\n{response.choices[0].message.content}"
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            # Continue to next model if this one fails
            continue
    
    # If all models failed, return error with helpful message
    error_msg = (
        f"LLaMA API Error: All attempted models failed.\n"
        f"Last error ({models_to_try[-1]}): {last_error}\n\n"
        f"Please check:\n"
        f"1. Your API key is valid at https://console.groq.com/\n"
        f"2. Available models at https://console.groq.com/docs/models\n"
        f"3. Your account has access to the requested models"
    )
    return generate_fallback_answer(query, retrieved_chunks, context, error=error_msg)


def generate_fallback_answer(query, chunks, formatted_context, error=None):
    """
    Generate a structured answer when LLM is not available
    This ensures users still get useful output for testing/development
    """
    if error:
        header = f"⚠️ LLM Error: {error}\n\n"
    else:
        header = "ℹ️ LLM not configured. Showing retrieved context in structured format.\n\n"
    
    # Try to extract key information from chunks
    answer = header
    answer += f"**Query:** {query}\n\n"
    
    # Generate a simple extractive summary
    if chunks:
        answer += "**Summary of Retrieved Information:**\n\n"
        
        # Extract key sentences from each chunk (first 2-3 sentences)
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            cleaned = " ".join(chunk.split())
            # Extract first few sentences as summary
            sentences = cleaned.split('. ')
            if len(sentences) > 3:
                summary = '. '.join(sentences[:3]) + '.'
            else:
                summary = cleaned
            
            # Truncate if still too long
            if len(summary) > 300:
                summary = summary[:300] + "..."
            
            summaries.append(f"**Source {i}:** {summary}")
        
        answer += "\n\n".join(summaries)
        answer += "\n\n"
    
    answer += "**Detailed Retrieved Information:**\n\n"
    
    for i, chunk in enumerate(chunks, 1):
        cleaned = " ".join(chunk.split())
        # Truncate very long chunks for readability
        if len(cleaned) > 400:
            cleaned = cleaned[:400] + "..."
        answer += f"**Source {i}:**\n{cleaned}\n\n"
    
    answer += "\n---\n\n"
    answer += "**Note:** To get AI-generated answers, please:\n"
    answer += "1. Set your GROQ_API_KEY environment variable\n"
    answer += "2. Or configure another LLM provider in the code\n"
    answer += "3. The system is currently showing raw retrieved context\n"
    
    return answer

# ============================================
# EXPERIMENTAL PIPELINE RUNNER
# ============================================

def run_experiment(query, top_k=5):
    """
    Run query across all experimental configurations
    Returns results for comparative analysis
    """
    results = {}
    
    for pipeline_name, cfg in EXPERIMENTAL_PIPELINES.items():
        print(f"\n[EXPERIMENT] Testing Pipeline: {pipeline_name}")
        
        # Retrieval phase
        retrieved_docs, metas = retrieve(pipeline_name, query, top_k)
        
        # Generation phase
        answer = generate_answer(query, retrieved_docs)
        
        results[pipeline_name] = {
            "config": cfg,
            "retrieved_docs": retrieved_docs,
            "metadata": metas,
            "answer": answer
        }
    
    return results

# ============================================
# HELPER FUNCTIONS FOR STREAMLIT
# ============================================

def diagnose_llm_setup():
    """
    Diagnose LLM setup issues and return helpful error messages
    """
    issues = []
    suggestions = []
    
    # Check if Groq library is installed
    if Groq is None:
        issues.append("❌ 'groq' package is not installed")
        suggestions.append("Install with: pip install groq")
    else:
        issues.append("✅ 'groq' package is installed")
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        issues.append("❌ GROQ_API_KEY environment variable is not set")
        suggestions.append("Set it via: export GROQ_API_KEY='your-key-here' or in Streamlit UI")
    else:
        issues.append(f"✅ GROQ_API_KEY is set (starts with: {api_key[:7]}...)")
        
        # Check API key format
        if not api_key.startswith('gsk_'):
            issues.append(f"⚠️ API key format may be invalid (should start with 'gsk_')")
            suggestions.append("Verify your API key at https://console.groq.com/")
        else:
            issues.append("✅ API key format looks correct")
            
            # Try to initialize client
            try:
                client = Groq(api_key=api_key)
                issues.append("✅ Groq client initialized successfully")
            except Exception as e:
                issues.append(f"❌ Failed to initialize client: {str(e)}")
                suggestions.append("Check your API key is valid and not expired")
    
    return {
        "issues": issues,
        "suggestions": suggestions,
        "ready": get_groq_client() is not None
    }

def get_system_status():
    """Check system initialization status"""
    return {
        "embedders_loaded": len(EMBEDDERS) > 0,
        "databases_ready": len(DBS) > 0,
        "llm_configured": get_groq_client() is not None,
        "pipelines_count": len(EXPERIMENTAL_PIPELINES)
    }

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG EXPERIMENTAL FRAMEWORK")
    print("Controlled Experimentation for Pipeline Optimization")
    print("="*60)
    
    # Check if docs folder exists
    if os.path.exists(DOCS_FOLDER) and len(os.listdir(DOCS_FOLDER)) > 0:
        print(f"\n[INFO] Found documents in {DOCS_FOLDER}/")
        index_all_pipelines()
        print("\n[INFO] System ready for experimentation")
    else:
        print(f"\n[INFO] No documents found in {DOCS_FOLDER}/")
        print("[INFO] Please use the Streamlit interface to upload documents")
    
    print("[INFO] Run: streamlit run streamlit_app.py")
    