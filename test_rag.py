# import os
# from pathlib import Path
# import pickle
# from typing import List, Dict
# import json
#
# # Document processing libraries
# from docx import Document
# import PyPDF2
# import pandas as pd
#
# # Free embedding and vector store
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
#
# # Configuration
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50
# VECTOR_DB_PATH = "vector_store"
# METADATA_PATH = "metadata.pkl"
# CHUNKS_OUTPUT_FILE = "chunks_preview.txt"
#
#
# class DocumentProcessor:
#     """Process different types of documents"""
#
#     @staticmethod
#     def read_pdf(file_path: str) -> str:
#         """Extract text from PDF"""
#         text = ""
#         try:
#             with open(file_path, 'rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 for page in pdf_reader.pages:
#                     text += page.extract_text() + "\n"
#         except Exception as e:
#             print(f"⚠️  Error reading PDF {Path(file_path).name}: {str(e)}")
#         return text
#
#     @staticmethod
#     def read_docx(file_path: str) -> str:
#         """Extract text from Word document"""
#         text = ""
#         try:
#             doc = Document(file_path)
#             for paragraph in doc.paragraphs:
#                 text += paragraph.text + "\n"
#             # Also extract text from tables
#             for table in doc.tables:
#                 for row in table.rows:
#                     for cell in row.cells:
#                         text += cell.text + " "
#                 text += "\n"
#         except Exception as e:
#             # Try alternative methods for old .doc files
#             try:
#                 import textract
#                 text = textract.process(file_path).decode('utf-8')
#             except:
#                 try:
#                     import win32com.client
#                     word = win32com.client.Dispatch("Word.Application")
#                     word.visible = False
#                     doc = word.Documents.Open(file_path)
#                     text = doc.Content.Text
#                     doc.Close()
#                     word.Quit()
#                 except:
#                     pass  # Skip files that can't be read
#         return text
#
#     @staticmethod
#     def read_excel(file_path: str) -> str:
#         """Extract text from Excel file"""
#         text = ""
#         try:
#             df = pd.read_excel(file_path, sheet_name=None)
#             for sheet_name, sheet_data in df.items():
#                 text += f"\n=== Sheet: {sheet_name} ===\n"
#                 text += sheet_data.to_string() + "\n"
#         except Exception as e:
#             print(f"⚠️  Error reading Excel {Path(file_path).name}: {str(e)}")
#         return text
#
#     def process_file(self, file_path: str) -> tuple:
#         """Process a single file based on extension"""
#         ext = Path(file_path).suffix.lower()
#         text = ""
#         error = None
#
#         try:
#             if ext == '.pdf':
#                 text = self.read_pdf(file_path)
#             elif ext in ['.docx', '.doc']:
#                 text = self.read_docx(file_path)
#             elif ext in ['.xlsx', '.xls']:
#                 text = self.read_excel(file_path)
#         except Exception as e:
#             error = str(e)
#
#         return text, error
#
#
# class TextChunker:
#     """Split text into chunks"""
#
#     @staticmethod
#     def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
#                    overlap: int = CHUNK_OVERLAP) -> List[str]:
#         """Split text into overlapping chunks"""
#         chunks = []
#         start = 0
#         text_length = len(text)
#
#         while start < text_length:
#             end = start + chunk_size
#             chunk = text[start:end]
#
#             # Try to end at sentence boundary
#             if end < text_length:
#                 last_period = chunk.rfind('.')
#                 last_newline = chunk.rfind('\n')
#                 break_point = max(last_period, last_newline)
#
#                 if break_point > chunk_size * 0.5:
#                     chunk = chunk[:break_point + 1]
#                     end = start + break_point + 1
#
#             chunks.append(chunk.strip())
#             start = end - overlap
#
#         return [c for c in chunks if c]
#
#
# class VectorStore:
#     """Manage embeddings and vector search"""
#
#     def __init__(self):
#         print("📦 Loading embedding model (all-MiniLM-L6-v2)...")
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.index = None
#         self.chunks = []
#         self.metadata = []
#         print("✅ Model loaded!\n")
#
#     def create_embeddings(self, chunks: List[str], metadata: List[Dict]):
#         """Create embeddings for chunks"""
#         print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
#
#         # Create embeddings in batches
#         batch_size = 32
#         all_embeddings = []
#
#         for i in range(0, len(chunks), batch_size):
#             batch = chunks[i:i + batch_size]
#             batch_embeddings = self.model.encode(batch, show_progress_bar=False)
#             all_embeddings.append(batch_embeddings)
#             print(f"   Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
#
#         embeddings = np.vstack(all_embeddings)
#
#         # Create FAISS index
#         dimension = embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(dimension)
#         self.index.add(embeddings.astype('float32'))
#
#         self.chunks = chunks
#         self.metadata = metadata
#
#         print("✅ Embeddings created!\n")
#
#         # Save to disk
#         self.save()
#
#     def save(self):
#         """Save vector store to disk"""
#         os.makedirs(VECTOR_DB_PATH, exist_ok=True)
#         faiss.write_index(self.index, f"{VECTOR_DB_PATH}/index.faiss")
#
#         with open(METADATA_PATH, 'wb') as f:
#             pickle.dump({
#                 'chunks': self.chunks,
#                 'metadata': self.metadata
#             }, f)
#
#         print(f"💾 Vector store saved to '{VECTOR_DB_PATH}/' directory\n")
#
#     def load(self):
#         """Load vector store from disk"""
#         try:
#             self.index = faiss.read_index(f"{VECTOR_DB_PATH}/index.faiss")
#
#             with open(METADATA_PATH, 'rb') as f:
#                 data = pickle.load(f)
#                 self.chunks = data['chunks']
#                 self.metadata = data['metadata']
#
#             print(f"✅ Loaded {len(self.chunks)} chunks from disk\n")
#             return True
#         except:
#             print("⚠️  No existing vector store found\n")
#             return False
#
#     def search(self, query: str, k: int = 5) -> List[Dict]:
#         """Search for similar chunks"""
#         query_embedding = self.model.encode([query])
#         distances, indices = self.index.search(query_embedding.astype('float32'), k)
#
#         results = []
#         for idx, dist in zip(indices[0], distances[0]):
#             results.append({
#                 'chunk': self.chunks[idx],
#                 'metadata': self.metadata[idx],
#                 'distance': float(dist)
#             })
#
#         return results
#
#
# def save_chunks_preview(chunks: List[str], metadata: List[Dict], output_file: str = CHUNKS_OUTPUT_FILE):
#     """Save chunks to a text file for manual review"""
#     print(f"📝 Saving chunks preview to '{output_file}'...")
#
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write("=" * 80 + "\n")
#         f.write("CHUNKS PREVIEW - CONSTRUCTION PROJECT DOCUMENTS\n")
#         f.write("=" * 80 + "\n\n")
#
#         for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
#             f.write(f"\n{'=' * 80}\n")
#             f.write(f"CHUNK #{i}\n")
#             f.write(f"{'=' * 80}\n")
#             f.write(f"File: {meta['file_name']}\n")
#             f.write(f"Section: {meta['section']}\n")
#             f.write(f"Path: {meta['file_path']}\n")
#             f.write(f"Length: {len(chunk)} characters\n")
#             f.write(f"{'-' * 80}\n")
#             f.write(f"{chunk}\n")
#
#             # Only save first 100 chunks to file (to avoid huge files)
#             if i >= 100:
#                 f.write(f"\n\n... and {len(chunks) - 100} more chunks (showing first 100)\n")
#                 break
#
#     print(f"✅ Chunks preview saved!\n")
#
#
# def process_project_folder(folder_path: str):
#     """Process all documents in the project folder"""
#
#     print("=" * 80)
#     print("STARTING DOCUMENT PROCESSING")
#     print("=" * 80 + "\n")
#
#     processor = DocumentProcessor()
#     chunker = TextChunker()
#
#     all_chunks = []
#     all_metadata = []
#
#     # Collect all files
#     print("📂 Scanning folder structure...")
#     files = []
#     for root, dirs, filenames in os.walk(folder_path):
#         for filename in filenames:
#             if filename.startswith('~$'):  # Skip temp files
#                 continue
#             file_path = os.path.join(root, filename)
#             ext = Path(file_path).suffix.lower()
#             if ext in ['.pdf', '.docx', '.doc', '.xlsx', '.xls']:
#                 files.append(file_path)
#
#     total_files = len(files)
#     print(f"✅ Found {total_files} documents to process\n")
#
#     processed_count = 0
#     error_count = 0
#     error_files = []
#
#     print("🔄 Processing documents...\n")
#
#     for idx, file_path in enumerate(files, 1):
#         file_name = Path(file_path).name
#         print(f"[{idx}/{total_files}] Processing: {file_name}")
#
#         # Process file
#         text, error = processor.process_file(file_path)
#
#         if error:
#             error_count += 1
#             error_files.append((file_name, error))
#             print(f"    ⚠️  Error: {error[:100]}")
#
#         if text and len(text.strip()) > 50:
#             # Chunk text
#             chunks = chunker.chunk_text(text)
#
#             # Add metadata
#             relative_path = os.path.relpath(file_path, folder_path)
#             for chunk in chunks:
#                 all_chunks.append(chunk)
#                 all_metadata.append({
#                     'file_name': file_name,
#                     'file_path': relative_path,
#                     'section': Path(file_path).parent.name
#                 })
#             processed_count += 1
#             print(f"    ✅ Created {len(chunks)} chunks")
#         else:
#             print(f"    ⏭️  Skipped (insufficient text)")
#
#     print("\n" + "=" * 80)
#     print("PROCESSING SUMMARY")
#     print("=" * 80)
#     print(f"✅ Successfully Processed: {processed_count} files")
#     print(f"📄 Total Chunks Created: {len(all_chunks)}")
#     print(f"⚠️  Files with Errors: {error_count}")
#
#     if error_files:
#         print(f"\n⚠️  Error Files (first 5):")
#         for fname, err in error_files[:5]:
#             print(f"   • {fname}")
#         if len(error_files) > 5:
#             print(f"   ... and {len(error_files) - 5} more")
#
#     print("=" * 80 + "\n")
#
#     return all_chunks, all_metadata
#
#
# def test_search(vector_store: VectorStore, query: str):
#     """Test search functionality"""
#     print("=" * 80)
#     print(f"SEARCH QUERY: {query}")
#     print("=" * 80 + "\n")
#
#     results = vector_store.search(query, k=3)
#
#     for i, result in enumerate(results, 1):
#         print(f"RESULT #{i}")
#         print(f"Distance: {result['distance']:.4f}")
#         print(f"File: {result['metadata']['file_name']}")
#         print(f"Section: {result['metadata']['section']}")
#         print(f"\nChunk Preview (first 300 chars):")
#         print("-" * 80)
#         print(result['chunk'][:300] + "...")
#         print("\n" + "=" * 80 + "\n")
#
#
# def main():
#     """Main execution function"""
#
#     print("\n" + "=" * 80)
#     print("CONSTRUCTION PROJECT RAG SYSTEM - TEST MODE")
#     print("=" * 80 + "\n")
#
#     # Configuration
#     PROJECT_FOLDER = input("Enter project folder path: ").strip()
#
#     if not os.path.exists(PROJECT_FOLDER):
#         print(f"❌ Error: Folder '{PROJECT_FOLDER}' does not exist!")
#         return
#
#     print("\n" + "=" * 80)
#     print("MENU")
#     print("=" * 80)
#     print("1. Process documents (creates embeddings)")
#     print("2. Load existing embeddings and test search")
#     print("=" * 80)
#
#     choice = input("\nEnter your choice (1 or 2): ").strip()
#
#     vector_store = VectorStore()
#
#     if choice == "1":
#         # Process documents
#         chunks, metadata = process_project_folder(PROJECT_FOLDER)
#
#         if not chunks:
#             print("❌ No chunks created. Please check your documents.")
#             return
#
#         # Save chunks preview
#         save_chunks_preview(chunks, metadata)
#
#         # Create embeddings
#         vector_store.create_embeddings(chunks, metadata)
#
#         print("✅ Processing complete!")
#         print(f"\n📄 You can review chunks manually in '{CHUNKS_OUTPUT_FILE}'")
#
#         # Ask if user wants to test search
#         test = input("\n🔍 Do you want to test search? (y/n): ").strip().lower()
#         if test == 'y':
#             while True:
#                 query = input("\nEnter search query (or 'quit' to exit): ").strip()
#                 if query.lower() == 'quit':
#                     break
#                 test_search(vector_store, query)
#
#     elif choice == "2":
#         # Load existing embeddings
#         if not vector_store.load():
#             print("❌ No existing embeddings found. Please run option 1 first.")
#             return
#
#         print(f"📊 Vector store loaded with {len(vector_store.chunks)} chunks\n")
#
#         # Interactive search
#         while True:
#             query = input("Enter search query (or 'quit' to exit): ").strip()
#             if query.lower() == 'quit':
#                 break
#             test_search(vector_store, query)
#
#     else:
#         print("❌ Invalid choice!")
#
#     print("\n" + "=" * 80)
#     print("DONE!")
#     print("=" * 80 + "\n")
#
#
# if __name__ == "__main__":
#     main()