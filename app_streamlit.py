import streamlit as st
import os
from pathlib import Path
import pickle
from typing import List, Dict
import tempfile
import requests
import json
import zipfile
import shutil

# Vector store
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Google Gemini
import google.generativeai as genai

# Configuration
MODEL_NAME = "gemini-2.0-flash"


class VectorStore:
    """Manage embeddings and vector search"""

    def __init__(self, store_type: str = "documents"):
        self.store_type = store_type
        self.model = None
        self.index = None
        self.chunks = []
        self.metadata = []

    def load_model(self):
        """Load embedding model"""
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def download_from_url(self, url: str, file_type: str) -> bytes:
        """Download file from URL"""
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Error downloading {file_type}: {str(e)}")
            return None

    def load_documents_from_urls(self, index_url: str, metadata_url: str):
        """Load document vector store from URLs"""
        try:
            index_content = self.download_from_url(index_url, "document index")
            if not index_content:
                return False

            metadata_content = self.download_from_url(metadata_url, "document metadata")
            if not metadata_content:
                return False

            with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp_index:
                tmp_index.write(index_content)
                tmp_index_path = tmp_index.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_meta:
                tmp_meta.write(metadata_content)
                tmp_meta_path = tmp_meta.name

            self.index = faiss.read_index(tmp_index_path)

            with open(tmp_meta_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']

            os.unlink(tmp_index_path)
            os.unlink(tmp_meta_path)

            return True

        except Exception as e:
            st.error(f"Error loading document store: {str(e)}")
            return False

    def load_excel_from_urls(self, index_url: str, metadata_url: str):
        """Load Excel vector store from URLs"""
        try:
            index_content = self.download_from_url(index_url, "Excel index")
            if not index_content:
                return False

            metadata_content = self.download_from_url(metadata_url, "Excel metadata")
            if not metadata_content:
                return False

            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_index:
                tmp_index.write(index_content)
                tmp_index_path = tmp_index.name

            self.index = faiss.read_index(tmp_index_path)

            metadata_json = json.loads(metadata_content.decode('utf-8'))
            self.metadata = metadata_json
            self.chunks = [f"Sheet: {m['sheet']}, Rows: {m['row_start']}-{m['row_end']}"
                           for m in metadata_json]

            os.unlink(tmp_index_path)

            return True

        except Exception as e:
            st.error(f"Error loading Excel store: {str(e)}")
            return False

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        if self.model is None:
            self.load_model()

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({
                    'chunk': self.chunks[idx] if idx < len(self.chunks) else "",
                    'metadata': self.metadata[idx],
                    'distance': float(dist),
                    'similarity': 1 / (1 + float(dist)),
                    'source_type': self.store_type
                })

        return results


class UnifiedVectorStore:
    """Manages multiple vector stores (documents + Excel)"""

    def __init__(self):
        self.document_store = None
        self.excel_store = None
        self.csv_folder = None  # Path to downloaded CSV files

    def load_document_store(self, index_url: str, metadata_url: str):
        """Load document vector store"""
        self.document_store = VectorStore(store_type="documents")
        self.document_store.load_model()
        return self.document_store.load_documents_from_urls(index_url, metadata_url)

    def load_excel_store(self, index_url: str, metadata_url: str, csv_url: str = None):
        """Load Excel vector store with optional CSV data"""
        self.excel_store = VectorStore(store_type="excel")
        self.excel_store.load_model()

        success = self.excel_store.load_excel_from_urls(index_url, metadata_url)

        # Download and extract CSV files if URL provided
        if success and csv_url:
            try:
                with st.spinner("📦 Downloading CSV files for detailed data..."):
                    csv_content = self.excel_store.download_from_url(csv_url, "CSV archive")

                    if csv_content:
                        # Create temp directory for CSVs
                        self.csv_folder = tempfile.mkdtemp()

                        # Save zip file
                        zip_path = os.path.join(self.csv_folder, "csv_sheets.zip")
                        with open(zip_path, 'wb') as f:
                            f.write(csv_content)

                        # Extract zip
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(self.csv_folder)

                        # Remove zip file
                        os.remove(zip_path)

                        st.success("✅ CSV files loaded - full detail available!")
                    else:
                        st.warning("⚠️ Could not download CSV files - using metadata only")
            except Exception as e:
                st.warning(f"⚠️ CSV download failed: {str(e)} - using metadata only")

        return success

    def search_all(self, query: str, k: int = 5, excel_threshold: float = 0.4) -> List[Dict]:
        """
        🔥 PRIORITY SEARCH: Excel FIRST, then Documents
        """
        excel_results = []
        doc_results = []

        # 🎯 STEP 1: Search Excel FIRST
        if self.excel_store and self.excel_store.index:
            excel_results = self.excel_store.search(query, k=k)

            # 🔥 ENHANCE: Add rich content to Excel results
            for result in excel_results:
                result['chunk'] = self._load_excel_content(result['metadata'])

            # Filter for good Excel matches
            good_excel = [r for r in excel_results if r['similarity'] > excel_threshold]

            if good_excel:
                # Excel has good results! Return ONLY Excel data
                return good_excel[:k]

        # 🎯 STEP 2: Excel has no good results, search Documents
        if self.document_store and self.document_store.index:
            doc_results = self.document_store.search(query, k=k)

        # 🎯 STEP 3: Combine results with Excel priority
        all_results = excel_results + doc_results
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results[:k]

    def _load_excel_content(self, metadata: Dict) -> str:
        """
        Load actual Excel content from CSV files if available,
        otherwise generate from metadata
        """
        sheet = metadata.get('sheet', 'Unknown')
        row_start = metadata.get('row_start', 0)
        row_end = metadata.get('row_end', 0)

        # Try to load from CSV files if available
        if self.csv_folder:
            try:
                # Import pandas only when needed
                try:
                    import pandas as pd
                except ImportError:
                    st.warning("⚠️ pandas not installed - cannot load CSV details")
                    return self._generate_metadata_content(metadata)

                # Find CSV file - check multiple possible paths
                possible_paths = [
                    os.path.join(self.csv_folder, f"{sheet}.csv"),
                    os.path.join(self.csv_folder, "csv_sheets", f"{sheet}.csv"),
                ]

                csv_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        csv_path = path
                        break

                if csv_path:
                    df = pd.read_csv(csv_path, dtype=str, na_filter=False)
                    chunk_df = df.iloc[row_start:row_end]

                    # Format as readable text
                    text_parts = [f"📊 Excel Sheet: {sheet} (Rows {row_start}-{row_end})\n"]

                    for idx, row in chunk_df.iterrows():
                        row_parts = []
                        for col, val in row.items():
                            val_clean = str(val).strip()
                            if val_clean and val_clean.lower() not in ['nan', 'none', '']:
                                row_parts.append(f"{col}: {val_clean}")
                        if row_parts:
                            text_parts.append(" | ".join(row_parts))

                    return "\n".join(text_parts)
            except Exception as e:
                # Fall through to metadata-only approach
                pass

        # Fallback: Generate from metadata only
        return self._generate_metadata_content(metadata)

    def _generate_metadata_content(self, metadata: Dict) -> str:
        """Generate descriptive content from metadata when CSV not available"""
        sheet = metadata.get('sheet', 'Unknown')
        row_start = metadata.get('row_start', 0)
        row_end = metadata.get('row_end', 0)

        content_lines = [
            f"📊 Excel Cost Data - Sheet: {sheet}",
            f"Rows: {row_start} to {row_end}",
            "",
            "This section contains construction cost information including:",
            "- Item descriptions and specifications",
            "- Quantities and units of measurement",
            "- Labour, material, and plant costs",
            "- Total amounts and rates",
            "",
            f"💡 Note: Upload csv_sheets.zip for detailed cost amounts",
            f"Data Location: Sheet '{sheet}', Rows {row_start}-{row_end}",
        ]

        return "\n".join(content_lines)

    def is_loaded(self) -> bool:
        """Check if at least one store is loaded"""
        doc_loaded = self.document_store and self.document_store.index
        excel_loaded = self.excel_store and self.excel_store.index
        return doc_loaded or excel_loaded

    def get_stats(self) -> Dict:
        """Get statistics about loaded stores"""
        stats = {
            'documents_loaded': False,
            'excel_loaded': False,
            'total_doc_chunks': 0,
            'total_excel_chunks': 0,
            'total_chunks': 0
        }

        if self.document_store and self.document_store.index:
            stats['documents_loaded'] = True
            stats['total_doc_chunks'] = len(self.document_store.chunks)

        if self.excel_store and self.excel_store.index:
            stats['excel_loaded'] = True
            stats['total_excel_chunks'] = len(self.excel_store.metadata)

        stats['total_chunks'] = stats['total_doc_chunks'] + stats['total_excel_chunks']

        return stats


class GeminiLLM:
    """Interface for Google Gemini"""

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
        return self.api_key is not None and self.model is not None

    def generate(self, prompt: str, stream: bool = True):
        """Generate response"""
        if not self.is_configured():
            yield "⚠️ Please configure your Gemini API key."
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


class UnifiedRAGChatbot:
    """RAG chatbot that searches BOTH sources and picks best results"""

    def __init__(self, vector_store: UnifiedVectorStore, llm: GeminiLLM):
        self.vector_store = vector_store
        self.llm = llm

    def create_unified_prompt(self, query: str, context: List[Dict]) -> str:
        """Create prompt with context from both sources"""

        doc_context = []
        excel_context = []

        for c in context:
            if c.get('source_type') == 'documents':
                doc_context.append(c)
            elif c.get('source_type') == 'excel':
                excel_context.append(c)

        context_text = ""

        if doc_context:
            context_text += "📄 FROM PROJECT DOCUMENTS:\n"
            for i, c in enumerate(doc_context, 1):
                file_name = c['metadata'].get('file_name', 'Unknown')
                context_text += f"\n--- Document {i}: {file_name} ---\n"
                context_text += f"{c['chunk']}\n"

        if excel_context:
            context_text += "\n💰 FROM COST PLAN (EXCEL):\n"
            for i, c in enumerate(excel_context, 1):
                sheet_name = c['metadata'].get('sheet', 'Unknown')
                context_text += f"\n--- Excel Sheet {i}: {sheet_name} ---\n"
                context_text += f"{c['chunk']}\n"

        prompt = f"""You are a construction project assistant with access to both project documents and detailed cost data.

RULES:
1. Answer ONLY from the provided context
2. If not in context, say: "I cannot find this information in the available data."
3. NEVER make up information
4. Cite sources (document name or Excel sheet)
5. For costs/budgets: use Excel data
6. For specifications/requirements: use documents
7. You can combine information from BOTH sources when relevant

AVAILABLE CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

        return prompt

    def ask(self, query: str, num_sources: int = 5):
        """Process question using unified search across BOTH sources"""

        # This searches BOTH stores and returns best matches from either/both
        results = self.vector_store.search_all(query, k=num_sources)
        filtered_results = [r for r in results if r['similarity'] > 0.3]

        if not filtered_results:
            yield "⚠️ No relevant information found in the loaded data."
            return

        prompt = self.create_unified_prompt(query, filtered_results)

        for chunk in self.llm.generate(prompt, stream=True):
            yield chunk

        return filtered_results


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'unified_store' not in st.session_state:
        st.session_state.unified_store = UnifiedVectorStore()
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'sources' not in st.session_state:
        st.session_state.sources = {}
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'excel_threshold' not in st.session_state:
        st.session_state.excel_threshold = 0.4


def auto_load_data():
    """Auto-load data from secrets on first run"""
    if st.session_state.data_loaded:
        return True

    # Check if all required secrets exist
    required_secrets = ['GEMINI_API_KEY', 'VECTOR_INDEX_URL', 'VECTOR_METADATA_URL',
                        'EXCEL_INDEX_URL', 'EXCEL_METADATA_URL']

    if not all(key in st.secrets for key in required_secrets):
        st.error("❌ Missing required secrets. Please check your secrets.toml file.")
        return False

    with st.spinner("🚀 Loading data on startup... This may take a minute."):

        # Load Gemini
        api_key = st.secrets["GEMINI_API_KEY"]
        st.session_state.llm = GeminiLLM(api_key=api_key)

        # Load Documents
        with st.spinner("📄 Loading project documents..."):
            doc_success = st.session_state.unified_store.load_document_store(
                st.secrets["VECTOR_INDEX_URL"],
                st.secrets["VECTOR_METADATA_URL"]
            )

        # Load Excel (with optional CSV)
        with st.spinner("💰 Loading cost plan data..."):
            csv_url = st.secrets.get("EXCEL_CSV_URL", None)
            excel_success = st.session_state.unified_store.load_excel_store(
                st.secrets["EXCEL_INDEX_URL"],
                st.secrets["EXCEL_METADATA_URL"],
                csv_url
            )

        if doc_success or excel_success:
            st.session_state.chatbot = UnifiedRAGChatbot(
                st.session_state.unified_store,
                st.session_state.llm
            )
            st.session_state.data_loaded = True
            return True
        else:
            st.error("❌ Failed to load data sources")
            return False


def display_source_card(source: Dict, index: int):
    """Display source card"""
    similarity_percentage = source['similarity'] * 100
    source_type = source.get('source_type', 'documents')

    icon = "📄" if source_type == "documents" else "💰"

    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            if source_type == "documents":
                st.markdown(f"{icon} **{source['metadata'].get('file_name', 'Unknown')}**")
                st.caption(f"Section: {source['metadata'].get('section', 'N/A')}")
            else:
                st.markdown(f"{icon} **Excel Sheet: {source['metadata'].get('sheet', 'Unknown')}**")
                st.caption(f"Rows: {source['metadata'].get('row_start', '?')}-{source['metadata'].get('row_end', '?')}")

        with col2:
            st.metric("Match", f"{similarity_percentage:.1f}%")

        with st.expander("View Content"):
            st.text(source['chunk'][:500] + ("..." if len(source['chunk']) > 500 else ""))


def main():
    st.set_page_config(
        page_title="Construction Project Assistant",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    initialize_session_state()

    # Auto-load data on startup
    if not st.session_state.data_loaded:
        if not auto_load_data():
            st.stop()
        else:
            st.success("✅ Data loaded successfully!")
            st.rerun()

    # Sidebar (minimal - just for stats)
    with st.sidebar:
        st.title("📊 System Status")

        if st.session_state.unified_store.is_loaded():
            stats = st.session_state.unified_store.get_stats()

            if stats['documents_loaded']:
                st.success(f"✅ Documents: {stats['total_doc_chunks']} chunks")

            if stats['excel_loaded']:
                st.success(f"✅ Cost Plan: {stats['total_excel_chunks']} chunks")

            st.metric("Total Searchable Chunks", stats['total_chunks'])

            st.info("💡 **Search Priority:** Excel first (costs/budget), then Documents (specifications/details)")

        st.divider()

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            excel_threshold = st.slider(
                "Excel relevance threshold",
                min_value=0.3,
                max_value=0.7,
                value=0.4,
                step=0.05,
                help="If Excel results are above this threshold, only Excel is used. Lower = more sensitive to Excel data."
            )
            st.session_state.excel_threshold = excel_threshold

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.sources = {}
            st.rerun()

    # Main content
    st.title("🏗️ Construction Project Assistant")
    st.markdown("Ask me anything! I search **Excel cost data first**, then project documents if needed.")

    # Show search strategy
    with st.expander("ℹ️ How does search work?", expanded=False):
        st.markdown("""
        ### 🎯 Search Priority Strategy:

        1. **Excel First** 🥇
           - For cost, budget, quantity questions
           - If Excel has good matches (>40% relevance), only Excel is used

        2. **Documents Second** 🥈
           - If Excel has no good matches
           - For specifications, requirements, schedules

        3. **Combined** 🤝
           - If Excel has some matches but not great
           - System combines both with Excel priority

        **Examples:**
        - "What is the cost of scaffolding?" → Excel only ✅
        - "What are the safety requirements?" → Documents only ✅
        - "Does the budget align with requirements?" → Both sources 🤝
        """)

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and idx in st.session_state.sources:
                sources = st.session_state.sources[idx]
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        for i, source in enumerate(sources, 1):
                            display_source_card(source, i)

    # Chat input
    if prompt := st.chat_input("Ask about costs, specifications, schedules, or anything else..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Get threshold from session state
            threshold = st.session_state.get('excel_threshold', 0.4)

            # 🔥 PRIORITY SEARCH: Excel first, then documents
            sources = st.session_state.unified_store.search_all(prompt, k=5, excel_threshold=threshold)
            filtered_sources = [s for s in sources if s['similarity'] > 0.3]

            if not filtered_sources:
                response_text = "⚠️ I cannot find relevant information to answer your question."
                message_placeholder.markdown(response_text)
                full_response = response_text
            else:
                for chunk in st.session_state.chatbot.ask(prompt, num_sources=5):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                with st.expander(f"📚 Sources ({len(filtered_sources)})", expanded=False):
                    for i, source in enumerate(filtered_sources, 1):
                        display_source_card(source, i)

                msg_idx = len(st.session_state.messages)
                st.session_state.sources[msg_idx] = filtered_sources

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()