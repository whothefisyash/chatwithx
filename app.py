import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    GitLoader,
    YoutubeLoader,
    UnstructuredURLLoader
)
import tempfile
import subprocess
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please add it and restart the application.")
    st.stop()

# Set page config
st.set_page_config(page_title="Chat with X", page_icon="🤖", layout="wide")

# CSS to improve UI
st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
}
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

def initialize_gemini_model():
    """Initialize the Google Gemini model with API key from .env"""
    try:
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",  # Using Gemini 1.5 Flash model
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=2048
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        return None

def create_conversational_chain(vector_store):
    """Create a conversational chain with the vector store"""
    llm = initialize_gemini_model()
    if llm is None:
        return None
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )
    
    return conversation_chain

def process_pdf(pdf_file):
    """Process a PDF file and return documents"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        os.unlink(tmp_file_path)  # Delete temp file
        return documents
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        os.unlink(tmp_file_path)  # Ensure temp file is deleted
        return []

def process_github_repo(repo_url):
    """Clone a GitHub repository and process its content"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Extract repo details
            repo_pattern = r"github\.com/([^/]+)/([^/]+)"
            match = re.search(repo_pattern, repo_url)
            if not match:
                st.error("Invalid GitHub repository URL")
                return []
            
            repo_owner, repo_name = match.groups()
            clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"
            
            # Clone repo
            st.info(f"Cloning repository: {clone_url}")
            subprocess.run(["git", "clone", clone_url, tmp_dir], check=True)
            
            # Load repository files
            loader = GitLoader(
                repo_path=tmp_dir,
                branch="main",
                file_filter=lambda file_path: file_path.endswith((".py", ".md", ".txt", ".js", ".html", ".css", ".json"))
            )
            
            documents = loader.load()
            return documents
        except Exception as e:
            st.error(f"Error processing GitHub repository: {e}")
            return []

def process_youtube_video(video_url):
    """Process a YouTube video and return documents with better error handling"""
    try:
        # First attempt using YoutubeLoader from LangChain
        loader = YoutubeLoader.from_youtube_url(
            video_url, 
            add_video_info=True,
            language=["en"]
        )
        documents = loader.load()
        if documents:
            return documents
        else:
            st.warning("No transcript found using primary method. Trying alternative method...")
    except Exception as e:
        st.warning(f"Primary YouTube processing method failed: {e}")
    
    # Fallback method using youtube-transcript-api if available
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from langchain.schema import Document
        
        # Extract video ID from URL
        video_id = None
        if "watch?v=" in video_url:
            video_id = video_url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
            
        if not video_id:
            st.error("Could not extract video ID from URL")
            return []
            
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine transcript parts
        full_transcript = ""
        for part in transcript_list:
            full_transcript += part['text'] + " "
            
        # Create document
        document = Document(
            page_content=full_transcript,
            metadata={"source": video_url, "title": f"YouTube Video: {video_id}"}
        )
        
        return [document]
    except Exception as e:
        st.error(f"All YouTube processing methods failed. Error: {e}")
        st.info("Suggestion: Check if the video has captions/transcripts available.")
        return []

def process_web_content(url):
    """Process web content (like Substack or arXiv) and return documents"""
    try:
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error processing web content: {e}")
        return []

def process_documents(documents):
    """Split documents and create a vector store"""
    if not documents:
        st.error("No documents to process.")
        return None
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        st.error("No text chunks extracted from documents.")
        return None
    
    st.info(f"Created {len(chunks)} chunks of text from the documents.")
    
    # Create embeddings and vector store
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def handle_user_input(user_question):
    """Process user question and generate response"""
    if st.session_state.conversation is None:
        st.warning("Please load a data source first.")
        return
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.conversation({"question": user_question})
            ai_response = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"I encountered an error: {str(e)}"})

def display_chat_messages():
    """Display chat messages in the UI"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    init_session_state()
    
    st.title("Chat with X: Your Multi-Source AI Assistant")
    st.caption("LLM used Gemini 1.5 Flash ")
    
    # Sidebar for data source selection
    with st.sidebar:
        st.header("Choose Data Source")
        data_source = st.selectbox(
            "Select Source Type",
            ["PDF Document", "GitHub Repository", "YouTube Video", "Web Content (arXiv/Substack)"]
        )
        
        if data_source == "PDF Document":
            uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
            if uploaded_file is not None and st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    documents = process_pdf(uploaded_file)
                    if documents:
                        st.session_state.vector_store = process_documents(documents)
                        if st.session_state.vector_store:
                            st.session_state.conversation = create_conversational_chain(st.session_state.vector_store)
                            st.session_state.data_source = f"PDF: {uploaded_file.name}"
                            st.success(f"Processed {len(documents)} pages from the PDF.")
                            st.session_state.messages = []  # Clear chat history
        
        elif data_source == "GitHub Repository":
            repo_url = st.text_input("Enter GitHub Repository URL (e.g., https://github.com/username/repo)")
            if repo_url and st.button("Process Repository"):
                with st.spinner("Processing GitHub Repository..."):
                    documents = process_github_repo(repo_url)
                    if documents:
                        st.session_state.vector_store = process_documents(documents)
                        if st.session_state.vector_store:
                            st.session_state.conversation = create_conversational_chain(st.session_state.vector_store)
                            st.session_state.data_source = f"GitHub: {repo_url}"
                            st.success(f"Processed {len(documents)} files from the repository.")
                            st.session_state.messages = []  # Clear chat history
        
        elif data_source == "YouTube Video":
            video_url = st.text_input("Enter YouTube Video URL")
            if video_url and st.button("Process Video"):
                with st.spinner("Processing YouTube Video..."):
                    documents = process_youtube_video(video_url)
                    if documents:
                        st.session_state.vector_store = process_documents(documents)
                        if st.session_state.vector_store:
                            st.session_state.conversation = create_conversational_chain(st.session_state.vector_store)
                            st.session_state.data_source = f"YouTube: {video_url}"
                            st.success("Processed YouTube video transcript.")
                            st.session_state.messages = []  # Clear chat history
        
        elif data_source == "Web Content (arXiv/Substack)":
            web_url = st.text_input("Enter URL (arXiv paper or Substack article)")
            if web_url and st.button("Process Web Content"):
                with st.spinner("Processing Web Content..."):
                    documents = process_web_content(web_url)
                    if documents:
                        st.session_state.vector_store = process_documents(documents)
                        if st.session_state.vector_store:
                            st.session_state.conversation = create_conversational_chain(st.session_state.vector_store)
                            st.session_state.data_source = f"Web: {web_url}"
                            st.success("Processed web content.")
                            st.session_state.messages = []  # Clear chat history
        
        # Display current data source
        if st.session_state.data_source:
            st.info(f"Current data source: {st.session_state.data_source}")
            if st.button("Clear Data Source"):
                st.session_state.data_source = None
                st.session_state.vector_store = None
                st.session_state.conversation = None
                st.session_state.messages = []
                st.experimental_rerun()
    
    # Main chat interface
    if st.session_state.data_source:
        st.header(f"Chat with: {st.session_state.data_source}")
        display_chat_messages()
        
        user_question = st.chat_input("Ask a question about your data...")
        if user_question:
            handle_user_input(user_question)
    else:
        st.info("Please select and process a data source to begin chatting.")

if __name__ == "__main__":
    main()