import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="RAG Q/A with PDF uploads",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Simple RAG Q/A App")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.write(
    "- Enter your Groq API Key\n"
    "- Upload PDFs\n"
    "- Ask questions and see chat history"
)
api_key = st.sidebar.text_input("🔑 GROQ API KEY", type="password")
hf_token = os.getenv("HF_TOKEN", "")

if not api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to continue")
    st.stop()

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

# File uploader
uploaded_files = st.file_uploader(
    "📂 Choose PDF File(s)",
    type="pdf",
    accept_multiple_files=True
)

# Function to load PDFs safely
def load_pdf(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            pdf_path = tmp.name
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load {file.name}: {e}")
        return []

# Process PDFs
all_docs = []
for pdf in uploaded_files:
    docs = load_pdf(pdf)
    all_docs.extend(docs)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(all_docs) if all_docs else []

# Create or load Chroma vectorstore
vectorstore = None
if splits:
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_index")
    retriever = vectorstore.as_retriever()
else:
    retriever = None

# Setup history-aware retriever and QA chain
if retriever:
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, decide what to return."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant. Use the retrieved context to answer. "
         "If you don't know, say no. Keep it under three sentences.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Initialize chat history per PDF
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    # Select PDF to chat about
    pdf_names = [pdf.name for pdf in uploaded_files] if uploaded_files else []
    selected_pdf = st.selectbox("Select PDF to chat with:", pdf_names) if pdf_names else "default"
    if selected_pdf not in st.session_state.chat_history:
        st.session_state.chat_history[selected_pdf] = []

    # --- Sidebar Clear Chat Button ---
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history[selected_pdf] = []
        st.sidebar.success(f"Chat history for '{selected_pdf}' cleared!")

    # --- Developed by Bushra line ---
    st.sidebar.markdown("---")  # optional separator
    st.sidebar.markdown("👩‍💻 **Developed by Bushra**")

    # Chat input
    user_input = st.chat_input("💬 Ask a question about the selected PDF...")
    if user_input:
        with st.spinner("🤔 Thinking..."):
            result = rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history[selected_pdf]
            })
        # Save chat history
        st.session_state.chat_history[selected_pdf].append(("user", user_input))
        st.session_state.chat_history[selected_pdf].append(("assistant", result["answer"]))

    # Display chat in bubbles
    for role, msg in st.session_state.chat_history[selected_pdf]:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg)

    # --- Download Chat History as PDF ---
    chat_history = st.session_state.chat_history.get(selected_pdf, [])
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    for role, msg in chat_history:
        text = f"{role.upper()}: {msg}"
        c.drawString(50, y, text)
        y -= 20
        if y < 50:  # next page if space khatam ho jaye
            c.showPage()
            y = height - 50
    c.save()
    pdf_buffer.seek(0)

    st.download_button(
        label="⬇️ Download Chat History (PDF)",
        data=pdf_buffer,
        file_name=f"{selected_pdf}_chat_history.pdf",
        mime="application/pdf"
    )

else:
    st.info("📌 Please upload PDF files to start.")

    # --- Still show empty PDF download button ---
    empty_pdf = io.BytesIO()
    c = canvas.Canvas(empty_pdf, pagesize=letter)
    c.drawString(50, 750, "No chat history available.")
    c.save()
    empty_pdf.seek(0)

    st.download_button(
        label="⬇️ Download Chat History (PDF)",
        data=empty_pdf,
        file_name="empty_chat_history.pdf",
        mime="application/pdf"
    )

