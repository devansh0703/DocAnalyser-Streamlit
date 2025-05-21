import streamlit as st
import os
import tempfile
import json

# LangChain components
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document # To manually create Document objects for JSON


GOOGLE_API_KEY = "AIzaSyAsBIw0b-EyKGJQNyt-ob6Tq_vSlwhsuJA"

if not GOOGLE_API_KEY:
    st.error("🚨 Google API Key not found. Please set it in your .env file and restart.")
    st.stop()

# --- Helper Function for Custom JSON Loading ---
def load_json_recursively(data, source_filename, path_prefix=""):
    """
    Recursively extracts text from JSON data and creates LangChain Document objects.
    """
    docs = []
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            docs.extend(load_json_recursively(value, source_filename, current_path))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{path_prefix}[{i}]"
            docs.extend(load_json_recursively(item, source_filename, current_path))
    elif isinstance(data, str):
        # Heuristic: only consider strings of reasonable length as actual content
        if len(data.strip()) > 20: # Avoid very short strings (e.g., isolated keys, "true")
            docs.append(Document(page_content=data, metadata={"source": source_filename, "json_path": path_prefix}))
    # You could extend this to handle numbers or booleans if you want to stringify them
    return docs

# --- Core LangChain RAG Functions ---

@st.cache_resource(show_spinner="⚙️ Processing documents... This may take a while for large files.")
def load_and_process_documents(uploaded_files, chunk_size, chunk_overlap):
    """
    Loads various document types, splits them, creates embeddings, and builds a vector store.
    Returns the vector store and a list of successfully processed file names.
    """
    all_documents = []
    processed_file_names = []

    if not uploaded_files:
        return None, []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = None
        current_docs = []
        try:
            if file_extension == ".txt":
                loader = TextLoader(tmp_file_path, encoding="utf-8")
                current_docs = loader.load()
            elif file_extension == ".pdf":
                loader = PyPDFLoader(tmp_file_path)
                current_docs = loader.load()
            elif file_extension == ".csv":
                loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
                # CSVLoader loads each row as a document. Ensure 'source' metadata is set.
                loaded_csv_docs = loader.load()
                for doc in loaded_csv_docs: # Ensure source metadata is the filename
                    doc.metadata["source"] = uploaded_file.name
                current_docs = loaded_csv_docs
            elif file_extension == ".docx":
                loader = Docx2txtLoader(tmp_file_path)
                current_docs = loader.load()
            elif file_extension == ".json":
                try:
                    with open(tmp_file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    current_docs = load_json_recursively(json_data, uploaded_file.name)
                except json.JSONDecodeError:
                    st.warning(f"⚠️ Could not parse JSON from '{uploaded_file.name}'. Skipping.")
                except Exception as e:
                    st.warning(f"⚠️ Error processing JSON '{uploaded_file.name}': {e}. Skipping.")

            else:
                st.warning(f"⚠️ Unsupported file type: '{uploaded_file.name}' ({file_extension}). Skipping.")
                os.remove(tmp_file_path) # Clean up temp file
                continue # Skip to next file

            if current_docs:
                all_documents.extend(current_docs)
                processed_file_names.append(uploaded_file.name)
                st.write(f"📄 Successfully loaded '{uploaded_file.name}' ({len(current_docs)} parts).")
            else:
                st.write(f"📄 No content extracted from '{uploaded_file.name}'.")

        except Exception as e:
            st.error(f"❌ Error loading/processing '{uploaded_file.name}': {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path) # Ensure cleanup

    if not all_documents:
        st.warning("⚠️ No documents were successfully processed.")
        return None, []

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(all_documents)
    if not texts:
        st.error("❌ Failed to split documents into manageable chunks.")
        return None, processed_file_names
    st.write(f"Splitting complete: {len(all_documents)} document(s) split into {len(texts)} chunks.")

    # 3. Create Embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing embeddings: {e}. Check API key & network.")
        return None, processed_file_names

    # 4. Create Vector Store (FAISS)
    try:
        vector_store = FAISS.from_documents(texts, embeddings)
        st.success(f"✅ Knowledge base built successfully from {len(processed_file_names)} file(s)!")
        return vector_store, processed_file_names
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        return None, processed_file_names


@st.cache_resource(show_spinner="✨ Initializing QA Chain...")
def get_conversational_chain(temperature):
    """
    Creates the question-answering chain with a custom prompt and Gemini LLM.
    """
    prompt_template = """
    You are an AI assistant specialized in answering questions based on the provided context documents.
    Carefully analyze the context before formulating your answer.
    Your goal is to provide accurate, detailed, and helpful responses derived exclusively from this context.

    Context:
    {context}

    Question:
    {question}

    Based *only* on the context provided above, please answer the question.
    If the information required to answer the question is not present in the context, clearly state:
    "The information to answer this question is not available in the provided documents."
    Do not attempt to find answers outside the given context or use external knowledge.
    Do not make up information. Be precise.
    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY,
                                     temperature=temperature, convert_system_message_to_human=True)
    except Exception as e:
        st.error(f"❌ Error initializing Gemini LLM: {e}. Check API key & network.")
        return None

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlit UI ---
st.set_page_config(page_title="📚 Advanced Document Chatbot (Gemini RAG)", layout="wide", initial_sidebar_state="expanded")
st.title("📚 Advanced Document Chatbot with Gemini & LangChain")
st.markdown("""
Upload your documents (TXT, PDF, CSV, DOCX, JSON) to build a knowledge base, then ask questions about their content.
Use the sidebar to manage documents and configure settings.
""")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload documents and build a knowledge base to start chatting."}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "llm_temperature" not in st.session_state:
    st.session_state.llm_temperature = 0.2
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 4 # Default number of chunks to retrieve
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200


# --- Sidebar for File Management and Settings ---
with st.sidebar:
    st.header("🛠️ Setup & Configuration")

    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (TXT, PDF, CSV, DOCX, JSON)",
        type=["txt", "pdf", "csv", "docx", "json"],
        accept_multiple_files=True,
        help="Upload one or more documents to build your knowledge base."
    )

    st.subheader("2. Knowledge Base Controls")
    if st.button("🏗️ Build/Rebuild Knowledge Base", type="primary", use_container_width=True,
                  help="Processes all uploaded files to create or update the knowledge base."):
        if uploaded_files:
            with st.spinner("Building knowledge base... This might take a moment."):
                st.session_state.vector_store, st.session_state.processed_files = load_and_process_documents(
                    uploaded_files, st.session_state.chunk_size, st.session_state.chunk_overlap
                )
                if st.session_state.vector_store:
                    st.session_state.qa_chain = get_conversational_chain(st.session_state.llm_temperature)
                    if st.session_state.qa_chain:
                        st.session_state.rag_ready = True
                        st.session_state.messages = [{"role": "assistant", "content": "✅ Knowledge base ready! Ask me anything about your documents."}]
                        st.success("Knowledge base built and ready!")
                    else:
                        st.session_state.rag_ready = False
                        st.error("Failed to initialize QA chain after building vector store.")
                else:
                    st.session_state.rag_ready = False
                    # Error messages are handled within load_and_process_documents
        else:
            st.warning("⚠️ Please upload at least one document before building the knowledge base.")

    if st.session_state.processed_files:
        st.markdown("##### Processed Files:")
        for f_name in st.session_state.processed_files:
            st.markdown(f"- `{f_name}`")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True,
                  help="Removes all processed documents and resets the knowledge base."):
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        st.session_state.rag_ready = False
        st.session_state.processed_files = []
        st.session_state.messages = [{"role": "assistant", "content": "Knowledge base cleared. Upload new documents to start again."}]
        st.info("Knowledge base cleared.")
        # Clear the file uploader widget's state too
        # This is a bit of a hack, as Streamlit doesn't directly support clearing FileUploader programmatically
        # Re-rendering it with a new key can sometimes achieve this.
        # Or, simply inform the user they might need to re-select files if they want to rebuild.
        # For simplicity, we'll rely on the user re-uploading if they clear.


    st.subheader("3. RAG Settings")
    st.session_state.llm_temperature = st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0,
        value=st.session_state.llm_temperature, step=0.1,
        help="Controls randomness. Lower values are more deterministic."
    )
    st.session_state.retrieval_k = st.slider(
        "Chunks to Retrieve (k)", min_value=1, max_value=10,
        value=st.session_state.retrieval_k, step=1,
        help="Number of relevant text chunks to fetch for context."
    )
    
    st.subheader("4. Text Splitting Settings")
    st.session_state.chunk_size = st.slider(
        "Chunk Size", min_value=100, max_value=4000,
        value=st.session_state.chunk_size, step=100,
        help="Max characters per text chunk. Rebuild KB after changing."
    )
    st.session_state.chunk_overlap = st.slider(
        "Chunk Overlap", min_value=0, max_value=1000,
        value=st.session_state.chunk_overlap, step=50,
        help="Characters to overlap between chunks. Rebuild KB after changing."
    )
    if st.session_state.qa_chain and (st.session_state.qa_chain.llm_chain.llm.temperature != st.session_state.llm_temperature):
        st.session_state.qa_chain = get_conversational_chain(st.session_state.llm_temperature) # Rebuild chain if temp changes


# --- Main Chat Interface ---
if st.session_state.rag_ready:
    st.success(f"✅ RAG system ready, querying **{len(st.session_state.processed_files)}** document(s).")
elif st.session_state.processed_files: # KB built, but maybe chain failed
    st.warning("⚠️ Knowledge base processed, but QA chain is not ready. Check configurations or API key.")
else:
    st.info("ℹ️ Please upload documents and build the knowledge base using the sidebar.")


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
             st.markdown(f"<small>Context from: {', '.join(message['sources'])}</small>", unsafe_allow_html=True)


# Chat input
if user_question := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.rag_ready:
        st.error("🚨 Please build the knowledge base first using the sidebar.")
    elif not st.session_state.vector_store or not st.session_state.qa_chain:
        st.error("🚨 RAG components are not initialized. Please try rebuilding the knowledge base.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🧠 Thinking... Retrieving context and generating answer..."):
                try:
                    # 1. Retrieve relevant documents
                    retrieved_docs = st.session_state.vector_store.similarity_search(
                        user_question, k=st.session_state.retrieval_k
                    )

                    if not retrieved_docs:
                        response_text = "I couldn't find any relevant information for your query in the loaded documents."
                        sources = []
                    else:
                        # 2. Generate answer using the RAG chain
                        chain_input = {"input_documents": retrieved_docs, "question": user_question}
                        result = st.session_state.qa_chain(chain_input, return_only_outputs=True)
                        response_text = result["output_text"]
                        sources = sorted(list(set([doc.metadata.get("source", "Unknown source") for doc in retrieved_docs])))

                    message_placeholder.markdown(response_text)
                    if sources:
                        st.markdown(f"<small>Context from: {', '.join(sources)}</small>", unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources
                    })


                except Exception as e:
                    error_message = f"An error occurred during processing: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {str(e)}"})

# Add a button to clear chat history
if len(st.session_state.messages) > 1: # Show only if there's more than the initial greeting
    if st.button("🧹 Clear Chat History", use_container_width=True):
        initial_message = "Hello! Upload documents and build a knowledge base to start chatting."
        if st.session_state.rag_ready:
            initial_message = f"✅ Knowledge base ready! Ask me anything about your documents ({len(st.session_state.processed_files)} files)."
        elif st.session_state.processed_files:
             initial_message = f"⚠️ Knowledge base loaded ({len(st.session_state.processed_files)} files), but QA chain might not be ready. Check sidebar."

        st.session_state.messages = [{"role": "assistant", "content": initial_message}]
        st.rerun()
