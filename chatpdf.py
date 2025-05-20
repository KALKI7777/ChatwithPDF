import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_model():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    
    return model, prompt



def user_input(user_question):
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        
        # Load the vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Search for similar documents
        docs = new_db.similarity_search(user_question, k=4)  # Get top 4 most relevant chunks
        
        # Get the model and prompt
        model, prompt = get_conversational_model()
        
        # Extract the content from documents and create the context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Use the model directly with the prompt template
        response = model.invoke(prompt.format(context=context, question=user_question))
        
        # Return the response content as a string
        return response.content
    except Exception as e:
        # Handle errors gracefully
        print(f"Error in user_input: {str(e)}")
        return f"I encountered an error: {str(e)}. Please make sure you've uploaded and processed documents before asking questions."




def main():
    # Page configuration with a custom theme
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS for better styling with a more engaging color scheme
    st.markdown("""
    <style>
    /* Overall page styling */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(to right, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-align: center;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subheader styling */
    .subheader {
        font-size: 1.7rem;
        color: #E0F7FA;
        margin-bottom: 20px;
        font-weight: 600;
        border-bottom: 2px solid #00BCD4;
        padding-bottom: 8px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(to right, #00B4DB, #0083B0);
        color: white;
        border-radius: 30px;
        padding: 12px 24px;
        font-weight: bold;
        width: 100%;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(to right, #0083B0, #00B4DB);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Upload section styling */
    .upload-section {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat container styling */
    .chat-container {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Message styling */
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #1A237E, #283593);
        color: white;
        border-left: 5px solid #536DFE;
        margin-left: 20px;
        margin-right: 5px;
    }
    
    /* AI message styling */
    .bot-message {
        background: linear-gradient(135deg, #004D40, #00695C);
        color: white;
        border-left: 5px solid #00BFA5;
        margin-right: 20px;
        margin-left: 5px;
    }
    
    /* Form styling */
    .stForm {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 12px 20px;
    }
    
    /* Info message styling */
    .stAlert {
        background-color: rgba(33, 150, 243, 0.1);
        color: #E3F2FD;
        border: 1px solid rgba(33, 150, 243, 0.2);
        border-radius: 10px;
    }
    
    /* Success message styling */
    .element-container div[data-testid="stText"] p:has(span[style*="color:green"]) {
        background-color: rgba(76, 175, 80, 0.1);
        color: #E8F5E9;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    
    /* File uploader styling */
    .stFileUploader>div>button {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* Progress bar styling */
    .stProgress>div>div>div>div {
        background: linear-gradient(to right, #00B4DB, #0083B0);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<h1 class="main-header">📚 PDF Chat Assistant with Gemini AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Upload your PDF documents and chat with them using Google Gemini AI</p>', unsafe_allow_html=True)
    
    # Create two columns for the layout - chat on left (larger), upload on right
    col1, col2 = st.columns([2, 1])
    
    # Main chat interface on the left
    with col1:
        st.markdown('<h2 class="subheader">💬 Chat Interface</h2>', unsafe_allow_html=True)
        
        # Initialize chat history in session state if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history with more visible styling
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display each message in the chat history
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        # User messages with gradient background and proper line breaks
                        # Replace newlines with <br> tags to preserve formatting
                        formatted_content = message["content"].replace("\n", "<br>")
                        st.markdown(
                            f'''
                            <div style="background: linear-gradient(135deg, #1A237E, #283593); padding: 15px; border-radius: 15px; margin-bottom: 15px; border-left: 5px solid #536DFE; color: white; margin-left: 20px; margin-right: 5px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); white-space: pre-wrap;">
                                <strong>You:</strong><br>{formatted_content}
                            </div>
                            ''', 
                            unsafe_allow_html=True
                        )
                    else:
                        # AI messages with gradient background and proper line breaks
                        # Replace newlines with <br> tags to preserve formatting
                        formatted_content = message["content"].replace("\n", "<br>")
                        st.markdown(
                            f'''
                            <div style="background: linear-gradient(135deg, #004D40, #00695C); padding: 15px; border-radius: 15px; margin-bottom: 15px; border-left: 5px solid #00BFA5; color: white; margin-right: 20px; margin-left: 5px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); white-space: pre-wrap;">
                                <strong>AI:</strong><br>{formatted_content}
                            </div>
                            ''', 
                            unsafe_allow_html=True
                        )
            else:
                st.info("Upload and process documents, then ask questions to start the conversation.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Initialize session state variables if they don't exist
        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0  # This will be used to reset the input field
        
        # Function to handle form submission and process the question immediately
        def handle_submit():
            if st.session_state.current_question.strip():  # Only process if there's a non-empty question
                # Get the question
                question = st.session_state.current_question.strip()
                
                # Add user question to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                try:
                    # Process the question immediately
                    response = user_input(question)
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    # Handle errors
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}. Please make sure you've processed documents before asking questions."})
        
        # Create a form for better input handling
        with st.form(key="question_form", clear_on_submit=True):
            # User input with the dynamic key
            user_question = st.text_input(
                "Ask a question about your documents",
                placeholder="What would you like to know about the PDFs?",
                key="current_question"
            )
            
            # Submit button
            submit_button = st.form_submit_button("Send", on_click=handle_submit)
    
    # Document upload on the right
    with col2:
        st.markdown('<h2 class="subheader">📄 Document Upload</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            pdf_docs = st.file_uploader(
                "Upload your PDF files",
                accept_multiple_files=True,
                help="Select one or more PDF files to analyze"
            )
            
            if st.button("Process Documents"):
                if pdf_docs:
                    with st.spinner("Processing your documents..."):
                        # Progress bar for better UX
                        progress_bar = st.progress(0)
                        
                        # Step 1: Extract text
                        progress_bar.progress(25)
                        st.info("Extracting text from PDFs...")
                        raw_text = get_pdf_text(pdf_docs)
                        
                        # Step 2: Split into chunks
                        progress_bar.progress(50)
                        st.info("Splitting text into chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Step 3: Create vector store
                        progress_bar.progress(75)
                        st.info("Creating vector embeddings...")
                        get_vector_store(text_chunks)
                        
                        # Complete
                        progress_bar.progress(100)
                        st.success("✅ Documents processed successfully! You can now ask questions.")
                else:
                    st.error("Please upload at least one PDF document")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # We don't need the pending_question logic anymore since we process questions directly in handle_submit
        # This section is intentionally left empty to remove the old processing logic



if __name__ == "__main__":
    main()
