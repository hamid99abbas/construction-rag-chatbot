import streamlit as st
import os
from pathlib import Path
import pickle
from typing import List, Dict
import tempfile
import requests
from io import BytesIO

# Vector store
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Google Gemini
import google.generativeai as genai

# Configuration
MODEL_NAME = "gemini-2.0-flash"

# Vector Database URLs - UPDATE THESE WITH YOUR LINKS
VECTOR_DB_CONFIG = {
    "index_url": "",  # Will be set from secrets or user input
    "metadata_url": "",  # Will be set from secrets or user input
}


class VectorStore:
    """Manage embeddings and vector search"""

    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.metadata = []

    def load_model(self):
        """Load embedding model"""
        if self.model is None:
            with st.spinner("Loading embedding model..."):
                self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def download_from_url(self, url: str, file_type: str) -> bytes:
        """Download file from URL"""
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Error downloading {file_type}: {str(e)}")
            return None

    def load_from_urls(self, index_url: str, metadata_url: str):
        """Load vector store from URLs (Google Drive, Dropbox, etc.)"""
        try:
            # Download index file
            with st.spinner("Downloading vector index..."):
                index_content = self.download_from_url(index_url, "index.faiss")
                if not index_content:
                    return False

            # Download metadata file
            with st.spinner("Downloading metadata..."):
                metadata_content = self.download_from_url(metadata_url, "metadata.pkl")
                if not metadata_content:
                    return False

            # Save to temporary files and load
            with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp_index:
                tmp_index.write(index_content)
                tmp_index_path = tmp_index.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_meta:
                tmp_meta.write(metadata_content)
                tmp_meta_path = tmp_meta.name

            # Load the files
            self.index = faiss.read_index(tmp_index_path)

            with open(tmp_meta_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']

            # Clean up temp files
            os.unlink(tmp_index_path)
            os.unlink(tmp_meta_path)

            return True

        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False

    def load_from_files(self, index_file, metadata_file):
        """Load vector store from uploaded files"""
        try:
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp_index:
                tmp_index.write(index_file.read())
                tmp_index_path = tmp_index.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_meta:
                tmp_meta.write(metadata_file.read())
                tmp_meta_path = tmp_meta.name

            # Load the files
            self.index = faiss.read_index(tmp_index_path)

            with open(tmp_meta_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']

            # Clean up temp files
            os.unlink(tmp_index_path)
            os.unlink(tmp_meta_path)

            return True
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        if self.model is None:
            self.load_model()

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'chunk': self.chunks[idx],
                'metadata': self.metadata[idx],
                'distance': float(dist),
                'similarity': 1 / (1 + float(dist))
            })

        return results


class GeminiLLM:
    """Interface for Google Gemini Flash"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.model = None

        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(
                    model_name=MODEL_NAME,
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    }
                )
            except Exception as e:
                st.error(f"Error configuring Gemini: {str(e)}")

    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return self.api_key is not None and self.model is not None

    def generate(self, prompt: str, stream: bool = True):
        """Generate response from Gemini"""

        if not self.is_configured():
            yield "⚠️ Please configure your Gemini API key in the sidebar."
            return

        try:
            if stream:
                response = self.model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                response = self.model.generate_content(prompt)
                return response.text

        except Exception as e:
            yield f"⚠️ Error: {str(e)}"


class RAGChatbot:
    """RAG-based chatbot with strict context adherence"""

    def __init__(self, vector_store: VectorStore, llm: GeminiLLM):
        self.vector_store = vector_store
        self.llm = llm

    def create_strict_prompt(self, query: str, context: List[Dict]) -> str:
        """Create a strict prompt that enforces answering only from context"""

        context_text = ""
        for i, c in enumerate(context, 1):
            context_text += f"\n--- Source {i}: {c['metadata']['file_name']} ---\n"
            context_text += f"{c['chunk']}\n"

        prompt = f"""You are a helpful assistant for a construction project. You must follow these STRICT RULES:

RULES:
1. ONLY answer based on the provided context below
2. If the answer is NOT in the context, you MUST say: "I cannot find this information in the provided documents."
3. NEVER make up information or use external knowledge
4. Always cite the source document name when answering
5. Be specific and quote relevant parts from the context
6. If the context is unclear or incomplete, say so

CONTEXT FROM PROJECT DOCUMENTS:
{context_text}

QUESTION: {query}

ANSWER (following all rules above):"""

        return prompt

    def ask(self, query: str, num_sources: int = 5):
        """Process a question and stream the response"""

        results = self.vector_store.search(query, k=num_sources)
        filtered_results = [r for r in results if r['similarity'] > 0.3]

        if not filtered_results:
            yield "⚠️ No relevant information found in the documents for your query."
            return

        prompt = self.create_strict_prompt(query, filtered_results)

        for chunk in self.llm.generate(prompt, stream=True):
            yield chunk

        return filtered_results


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'sources' not in st.session_state:
        st.session_state.sources = {}


def display_source_card(source: Dict, index: int):
    """Display a source document card"""
    similarity_percentage = source['similarity'] * 100

    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**📄 {source['metadata']['file_name']}**")
            st.caption(f"Section: {source['metadata']['section']}")

        with col2:
            st.metric("Match", f"{similarity_percentage:.1f}%")

        with st.expander("View Content"):
            st.text(source['chunk'][:500] + ("..." if len(source['chunk']) > 500 else ""))


def main():
    st.set_page_config(
        page_title="Construction Project Assistant",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .stAlert { padding: 1rem; margin: 1rem 0; }
        .source-card { border: 1px solid #ddd; border-radius: 5px; padding: 1rem; margin: 0.5rem 0; }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")

        # API Key Input
        st.subheader("🔑 Gemini API Key")

        api_key = None
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.success("✅ API Key loaded from secrets")
        else:
            api_key = st.text_input(
                "Enter your Gemini API Key",
                type="password",
                help="Get your free API key from https://aistudio.google.com/apikey"
            )

        if api_key:
            if st.session_state.llm is None or st.session_state.llm.api_key != api_key:
                st.session_state.llm = GeminiLLM(api_key=api_key)
                if st.session_state.llm.is_configured():
                    st.success("✅ Gemini configured")
        else:
            st.info("👆 Please enter your Gemini API key")
            st.markdown("[Get Free API Key →](https://aistudio.google.com/apikey)")

        st.divider()

        # Vector Store Loading Options
        st.subheader("📊 Load Vector Database")

        # Check if URLs are configured in secrets
        use_urls = False
        index_url = ""
        metadata_url = ""

        if "VECTOR_INDEX_URL" in st.secrets and "VECTOR_METADATA_URL" in st.secrets:
            index_url = st.secrets["VECTOR_INDEX_URL"]
            metadata_url = st.secrets["VECTOR_METADATA_URL"]
            use_urls = True
            st.info("📡 Vector DB URLs configured in secrets")

        # Tab selection for loading method
        load_method = st.radio(
            "Choose loading method:",
            ["Auto-download from URLs", "Manual upload"],
            index=0 if use_urls else 1
        )

        if load_method == "Auto-download from URLs":
            st.info("📥 Download vector database from cloud storage")

            # Allow URL input if not in secrets
            if not use_urls:
                index_url = st.text_input(
                    "Index URL (index.faiss)",
                    placeholder="https://drive.google.com/uc?export=download&id=...",
                    help="Direct download link to index.faiss"
                )
                metadata_url = st.text_input(
                    "Metadata URL (metadata.pkl)",
                    placeholder="https://drive.google.com/uc?export=download&id=...",
                    help="Direct download link to metadata.pkl"
                )

            if index_url and metadata_url:
                if st.button("🌐 Download & Load Vector Store", type="primary"):
                    with st.spinner("Downloading from cloud..."):
                        vector_store = VectorStore()
                        vector_store.load_model()

                        if vector_store.load_from_urls(index_url, metadata_url):
                            st.session_state.vector_store = vector_store
                            st.session_state.chatbot = RAGChatbot(vector_store, st.session_state.llm)
                            st.success("✅ Vector store loaded from URLs!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to download vector store")
            else:
                st.warning("⚠️ Please provide both URLs")

            # Show guide for getting direct download links
            with st.expander("📖 How to get direct download links"):
                st.markdown("""
                ### Google Drive
                1. Upload files to Google Drive
                2. Right-click → Share → Anyone with link
                3. Get shareable link: `https://drive.google.com/file/d/FILE_ID/view`
                4. Convert to direct link: `https://drive.google.com/uc?export=download&id=FILE_ID`

                ### Dropbox
                1. Upload files to Dropbox
                2. Right-click → Share → Create link
                3. Change `?dl=0` to `?dl=1` at end of URL

                ### GitHub Releases
                1. Create a release in your repo
                2. Attach files as assets
                3. Use raw download URL
                """)

        else:  # Manual upload
            st.info("📤 Upload vector database files manually")

            index_file = st.file_uploader(
                "Upload index.faiss",
                type=['faiss'],
                help="The FAISS index file from vector_store folder"
            )

            metadata_file = st.file_uploader(
                "Upload metadata.pkl",
                type=['pkl'],
                help="The metadata pickle file"
            )

            if index_file and metadata_file:
                if st.button("🔄 Load Vector Store", type="primary"):
                    with st.spinner("Loading vector store..."):
                        vector_store = VectorStore()
                        vector_store.load_model()

                        if vector_store.load_from_files(index_file, metadata_file):
                            st.session_state.vector_store = vector_store
                            st.session_state.chatbot = RAGChatbot(vector_store, st.session_state.llm)
                            st.success("✅ Vector store loaded!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load vector store")

        # Display vector store stats
        if st.session_state.vector_store:
            st.success("✅ Vector store loaded")
            st.metric("Total Chunks", len(st.session_state.vector_store.chunks))

            unique_files = len(set([m['file_name'] for m in st.session_state.vector_store.metadata]))
            st.metric("Unique Files", unique_files)

        st.divider()

        # Settings
        st.subheader("⚙️ Search Settings")
        num_sources = st.slider("Number of sources", 3, 10, 5,
                                help="Number of document chunks to retrieve")

        st.divider()

        # Clear Chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.sources = {}
            st.rerun()

    # Main content
    st.title("🏗️ Construction Project Assistant")
    st.markdown("Ask questions about your project documents. I only answer based on the provided documents.")

    # Status checks
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar")
        st.info("👉 Get your free API key from [Google AI Studio](https://aistudio.google.com/apikey)")
        return

    if not st.session_state.llm or not st.session_state.llm.is_configured():
        st.warning("⚠️ Please configure your API key in the sidebar")
        return

    if not st.session_state.vector_store:
        st.info("ℹ️ Please load the vector store from the sidebar")
        st.markdown("""
        ### 📝 Two ways to load:
        1. **Auto-download**: Provide cloud storage URLs (Google Drive, Dropbox)
        2. **Manual upload**: Upload files directly
        """)
        return

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and idx in st.session_state.sources:
                sources = st.session_state.sources[idx]
                if sources:
                    with st.expander(f"📚 View {len(sources)} Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            display_source_card(source, i)

    # Chat input
    if prompt := st.chat_input("Ask about your project...", key="chat_input"):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            sources = st.session_state.vector_store.search(prompt, k=num_sources)
            filtered_sources = [s for s in sources if s['similarity'] > 0.3]

            if not filtered_sources:
                response_text = "⚠️ I cannot find relevant information in the provided documents to answer your question."
                message_placeholder.markdown(response_text)
                full_response = response_text
            else:
                for chunk in st.session_state.chatbot.ask(prompt, num_sources=num_sources):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                with st.expander(f"📚 View {len(filtered_sources)} Sources", expanded=False):
                    for i, source in enumerate(filtered_sources, 1):
                        display_source_card(source, i)

                msg_idx = len(st.session_state.messages)
                st.session_state.sources[msg_idx] = filtered_sources

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()