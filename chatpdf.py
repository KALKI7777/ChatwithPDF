import streamlit as st
from PyPDF2 import PdfReader # reads pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter # splits text
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # embeds text
import google.generativeai as genai 
from langchain_community.vectorstores import FAISS # creates vector store
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain # loads qa chain
from langchain.prompts import PromptTemplate # loads prompt
from dotenv import load_dotenv # loads api key

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """
    Function to extract text from pdfs
    Parameters:
        pdf_docs (list): List of pdf documents
    Returns:
        text (str): Text extracted from pdfs
    """
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    """
    Splits the given text into chunks of specified size with overlap.

    Parameters:
        text (str): The text to be split into chunks.

    Returns:
        List[str]: A list of text chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Create a vector store from the given text chunks.

    Parameters:
        text_chunks (list): A list of text chunks.

    Returns:
        None
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    """
    This function takes a user question as input and returns the best answer based on the input text chunks.

    Parameters:
        user_question (str): The user's question.

    Returns:
        None
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



def main():
    # Configure the page with custom theme and layout
    st.set_page_config(
        page_title="PDF Assistant Pro", 
        page_icon="📚", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize session state for user input processing
    if 'user_input_processed' not in st.session_state:
        st.session_state.user_input_processed = False
    
    # Reset the processed flag when the page loads
    st.session_state.user_input_processed = False
    
    # Function to handle question submission
    def handle_question_submission(question):
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M")
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": current_time
        })
        
        # Process the question
        try:
            # Get the answer
            embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(question)
            
            chain = get_conversational_chain()
            response = chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["output_text"],
                "timestamp": current_time
            })
            
        except Exception as e:
            # Add error message to chat history
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_message,
                "timestamp": current_time
            })
        
        # Mark input as processed
        st.session_state.user_input_processed = True
        
        # Clear the input box
        st.session_state.user_input = ""
    
    # Callback for text input
    def on_input_change():
        if st.session_state.user_input and not st.session_state.user_input_processed:
            handle_question_submission(st.session_state.user_input)
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #4527A0;
            --secondary-color: #7E57C2;
            --accent-color: #FFD54F;
            --text-color: #212121;
            --background-color: #F5F7F9;
            --card-background: #FFFFFF;
        }
        
        /* Header styling */
        .header {
            padding: 1.5rem 0;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 0rem;
        }
        
        .header h1 {
            color: var(--primary-color);
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        /* Section headings */
        h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        
        /* Remove extra padding from containers */
        .stVerticalBlock {
            gap: 0.5rem !important;
        }
        
        /* Message styling */
        .message {
            margin-bottom: 1rem;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            max-width: 80%;
            position: relative;
            word-wrap: break-word;
        }
        
        .user-message {
            background-color: var(--primary-color);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 0;
        }
        
        .assistant-message {
            background-color: white;
            border: 1px solid #e0e0e0;
            margin-right: auto;
            border-bottom-left-radius: 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            color: rgb(69, 39, 160);
        }
        
        /* Timestamp styling */
        .timestamp {
            font-size: 0.7rem;
            opacity: 0.7;
            text-align: right;
            margin-top: 0.3rem;
        }
        
        /* Input area styling */
        .input-area {
            display: flex;
            gap: 0.5rem;
        }
        
        /* Welcome message */
        .welcome-message {
            text-align: center;
            color: #666;
            padding: 2rem;
            margin: 2rem 0;
            background-color: #f9f9f9;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        
        /* Welcome message details */
        .welcome-message h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        
        .welcome-message p {
            margin: 0.25rem 0;
        }
        
        /* File upload area */
        .file-upload {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: var(--secondary-color);
        }
        
        /* Example questions */
        .example-question {
            background-color: rgba(126, 87, 194, 0.1);
            padding: 0.5rem 1rem;
            border-radius: 5px;
            margin-bottom: 0.5rem;
            cursor: pointer;
        }
        
        .example-question:hover {
            background-color: rgba(126, 87, 194, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown("<div class='header'><h1>📚 PDF Assistant Pro</h1><p>Upload PDFs and chat with your documents</p></div>", unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # File upload section
        st.markdown("<h3>Upload Documents</h3>", unsafe_allow_html=True)
        
        with st.container():
            pdf_docs = st.file_uploader(
                "Upload PDF files", 
                accept_multiple_files=True,
                type=['pdf'],
                help="Upload one or more PDF documents to analyze"
            )
            
            # Process button
            process_btn = st.button("Process Documents", type="primary", key="process_btn")
            
            # Display uploaded files
            if pdf_docs:
                st.write(f"**{len(pdf_docs)}** documents ready")
                for pdf in pdf_docs:
                    st.write(f"📄 {pdf.name}")
            
            if process_btn and pdf_docs:
                with st.spinner("Processing documents..."):
                    # Process the PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    # Update progress
                    progress_bar.progress(30)
                    st.info("Splitting text...")
                    text_chunks = get_text_chunks(raw_text)
                    
                    progress_bar.progress(60)
                    st.info("Creating embeddings...")
                    get_vector_store(text_chunks)
                    
                    progress_bar.progress(100)
                    st.success("✅ Processing complete!")
                    
                    # Add system message to chat
                    from datetime import datetime
                    current_time = datetime.now().strftime("%H:%M")
                    
                    system_message = f"I've processed {len(pdf_docs)} documents with {len(text_chunks)} text chunks. You can now ask questions about them."
                    
                    if not any(msg.get("content") == system_message for msg in st.session_state.chat_history if msg.get("role") == "assistant"):
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": system_message,
                            "timestamp": current_time
                        })
        
    with col1:
        # Chat heading
        st.markdown("<h3>Chat with your Documents</h3>", unsafe_allow_html=True)
        
        # Welcome message if no chat history
        if not st.session_state.chat_history:
            st.markdown("""
            <div class='welcome-message'>
                <h3>Welcome to PDF Assistant Pro!</h3>
                <p>Upload your PDF documents and ask questions about them.</p>
                <p>Your conversation will appear here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display all messages in the chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class='message user-message'>
                        {message["content"]}
                        <div class='timestamp'>{message["timestamp"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='message assistant-message'>
                        {message["content"]}
                        <div class='timestamp'>{message["timestamp"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat History", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Chat input
        st.markdown("<div class='input-area'>", unsafe_allow_html=True)
        user_input = st.text_input(
            "Ask a question about your documents:",
            key="user_input",
            on_change=on_input_change
        )
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()