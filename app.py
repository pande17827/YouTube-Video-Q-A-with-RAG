import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma, Weaviate, Qdrant
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re
import os
import asyncio
import getpass

# Fix for the event loop error
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set up the app
st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
            color: white;
        }
        h1 {
            color: #343a40;
            border-bottom: 2px solid #6c757d;
            padding-bottom: 10px;
        }
        .stButton>button {
            background-color: #28a745;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 10px;
        }
        .stTextArea>div>div>textarea {
            border-radius: 5px;
            padding: 10px;
        }
        .success {
            color: #28a745;
            font-weight: bold;
        }
        .error {
            color: #dc3545;
            font-weight: bold;
        }
        .thumbnail {
            max-width: 100%;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .settings-section {
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 5px;
            background-color: rgba(255,255,255,0.1);
        }
        .model-option {
            margin-bottom: 10px;
        }
        [data-testid="stRadio"] > div {
            flex-direction: row !important;
            gap: 20px;
        }
        .connection-info {
            font-size: 0.8em;
            color: #6c757d;
            margin-top: -10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🎬 YouTube Video Q&A with RAG")
st.markdown("""
    Ask questions about any YouTube video! This app uses Retrieval-Augmented Generation (RAG) to provide answers based on the video's transcript.
""")

# Initialize session state for vector store connections
if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Hugging Face API token input
    with st.expander("🔑 API Configuration", expanded=True):
        hf_token = st.text_input("Hugging Face API Token", type="password", 
                                help="Enter your Hugging Face API token to use the models")
    
    # Generation Model Selection
    with st.expander("🤖 Generation Model", expanded=True):
        # Model selection method
        model_selection_method = st.radio(
            "Generation model source:",
            ["Use preset model", "Use custom model"],
            index=0,
            horizontal=True,
            key="gen_model_selector"
        )
        
        if model_selection_method == "Use preset model":
            # Predefined LLM options
            llm_options = {
                "Qwen/Qwen1.5-32B": "Qwen/Qwen1.5-32B",
                "Mistral-7B": "mistralai/Mistral-7B-v0.1",
                "FLAN-T5 Large": "google/flan-t5-large",
                "BART Large CNN": "facebook/bart-large-cnn",
                "GPT-2": "gpt2"
            }
            selected_llm = st.selectbox(
                "Select generation model:",
                list(llm_options.keys()),
                index=0,
                key="preset_llm"
            )
            llm_model_name = llm_options[selected_llm]
        else:
            llm_model_name = st.text_input(
                "Enter custom generation model:",
                placeholder="username/model-name",
                value="Qwen/QwQ-32B",
                key="custom_llm",
                help="Example: 'Qwen/QwQ-32B' or 'mistralai/Mixtral-8x7B-Instruct-v0.1'"
            )
        
        # LLM parameters
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Controls randomness (lower = more deterministic)"
            )
        with col2:
            max_length = st.slider(
                "Max Length",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help="Maximum length of generated response"
            )
    
    # Embedding Model Selection
    with st.expander("🔍 Embedding Model", expanded=True):
        # Embedding selection method
        embedding_selection_method = st.radio(
            "Embedding model source:",
            ["Use preset model", "Use custom model"],
            index=0,
            horizontal=True,
            key="emb_model_selector"
        )
        
        if embedding_selection_method == "Use preset model":
            # Predefined embedding options
            embedding_options = {
                "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
                "bge-small-en": "BAAI/bge-small-en-v1.5",
                "e5-small-v2": "intfloat/e5-small-v2",
                "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
            }
            selected_embedding = st.selectbox(
                "Select embedding model:",
                list(embedding_options.keys()),
                index=0,
                key="preset_embedding"
            )
            embedding_model_name = embedding_options[selected_embedding]
        else:
            embedding_model_name = st.text_input(
                "Enter custom embedding model:",
                placeholder="username/model-name",
                value="intfloat/e5-small-v2",
                key="custom_embedding",
                help="Example: 'intfloat/e5-small-v2' or 'BAAI/bge-base-en-v1.5'"
            )
    
    # Vector Store Selection
    with st.expander("🗄️ Vector Database", expanded=True):
        # Vector store selection method
        vectorstore_selection_method = st.radio(
            "Vector store type:",
            ["Use preset configuration", "Custom configuration"],
            index=0,
            horizontal=True,
            key="vectorstore_selector"
        )
        
        if vectorstore_selection_method == "Use preset configuration":
            # Predefined vector store options
            vectorstore_options = {
                "FAISS (Local)": "faiss",
                "Chroma (Local)": "chroma",
                "Chroma (Persistent)": "chroma_persistent",
                "Weaviate (Local)": "weaviate_local",
                "Pinecone (Cloud)": "pinecone",
                "Qdrant (Local)": "qdrant_local",
                "Qdrant (Cloud)": "qdrant_cloud"
            }
            selected_vectorstore = st.selectbox(
                "Select vector database:",
                list(vectorstore_options.keys()),
                index=0,
                key="preset_vectorstore"
            )
            vectorstore_type = vectorstore_options[selected_vectorstore]
            
            # Additional parameters for specific stores
            if vectorstore_type == "chroma_persistent":
                persist_directory = st.text_input(
                    "Persist directory:",
                    value="./chroma_db",
                    help="Directory to store the Chroma database"
                )
            elif vectorstore_type in ["pinecone", "qdrant_cloud"]:
                st.info("Please configure connection details below")
                
        else:
            # Custom vector store configuration
            vectorstore_type = st.selectbox(
                "Select database type:",
                ["faiss", "chroma", "chroma_persistent", "weaviate", "pinecone", "qdrant"],
                index=0,
                key="custom_vectorstore_type"
            )
            
            if vectorstore_type == "chroma_persistent":
                persist_directory = st.text_input(
                    "Persist directory:",
                    value="./chroma_db",
                    help="Directory to store the Chroma database"
                )
            
            if vectorstore_type in ["weaviate", "pinecone", "qdrant"]:
                st.subheader("Connection Configuration")
                
                if vectorstore_type == "weaviate":
                    weaviate_url = st.text_input(
                        "Weaviate URL:",
                        value="http://localhost:8080",
                        help="URL of your Weaviate instance"
                    )
                    weaviate_api_key = st.text_input(
                        "Weaviate API Key (optional):",
                        type="password",
                        help="Leave empty if no authentication required"
                    )
                    weaviate_index = st.text_input(
                        "Index/Collection Name:",
                        value="YouTubeVideos",
                        help="Name for your Weaviate collection"
                    )
                
                elif vectorstore_type == "pinecone":
                    pinecone_api_key = st.text_input(
                        "Pinecone API Key:",
                        type="password",
                        help="Your Pinecone API key"
                    )
                    pinecone_env = st.text_input(
                        "Pinecone Environment:",
                        value="us-west1-gcp",
                        help="e.g. 'us-west1-gcp'"
                    )
                    pinecone_index = st.text_input(
                        "Pinecone Index Name:",
                        value="youtube-videos",
                        help="Name for your Pinecone index"
                    )
                
                elif vectorstore_type == "qdrant":
                    qdrant_url = st.text_input(
                        "Qdrant URL:",
                        value="http://localhost:6333",
                        help="URL of your Qdrant instance"
                    )
                    qdrant_api_key = st.text_input(
                        "Qdrant API Key (optional):",
                        type="password",
                        help="Leave empty if no authentication required"
                    )
                    qdrant_collection = st.text_input(
                        "Collection Name:",
                        value="youtube_videos",
                        help="Name for your Qdrant collection"
                    )
    
    # Processing settings
    with st.expander("⚡ Processing Settings", expanded=True):
        # Language selection
        language_options = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Hindi": "hi",
            "Auto-detect": None
        }
        
        selected_language = st.selectbox(
            "Transcript Language:",
            list(language_options.keys()),
            index=0,
            key="language"
        )
        
        # Chunk settings
        chunk_size = st.slider(
            "Chunk Size:", 
            500, 
            2000, 
            1000,
            key="chunk_size"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap:", 
            0, 
            500, 
            200,
            key="chunk_overlap"
        )
        
        k = st.slider(
            "Documents to Retrieve:", 
            1, 
            10, 
            4,
            key="retrieve_k"
        )

def create_vector_store(docs, embeddings, vectorstore_type, **kwargs):
    """Create vector store based on selected type"""
    if vectorstore_type == "faiss":
        return FAISS.from_documents(docs, embeddings)
    
    elif vectorstore_type in ["chroma", "chroma_persistent"]:
        persist_directory = kwargs.get("persist_directory", "./chroma_db")
        if vectorstore_type == "chroma_persistent":
            return Chroma.from_documents(
                docs, 
                embeddings, 
                persist_directory=persist_directory
            )
        return Chroma.from_documents(docs, embeddings)
    
    elif vectorstore_type == "weaviate_local":
        return Weaviate.from_documents(
            docs,
            embeddings,
            weaviate_url="http://localhost:8080",
            index_name="YouTubeVideos"
        )
    
    elif vectorstore_type == "weaviate":
        return Weaviate.from_documents(
            docs,
            embeddings,
            weaviate_url=kwargs.get("weaviate_url"),
            by_text=False,
            index_name=kwargs.get("weaviate_index"),
            api_key=kwargs.get("weaviate_api_key")
        )
    
    elif vectorstore_type == "pinecone":
        try:
            import pinecone
            pinecone.init(
                api_key=kwargs.get("pinecone_api_key"),
                environment=kwargs.get("pinecone_env")
            )
            return Pinecone.from_documents(
                docs,
                embeddings,
                index_name=kwargs.get("pinecone_index")
            )
        except ImportError:
            st.error("Pinecone client not installed. Please install with: pip install pinecone-client")
            return None
    
    elif vectorstore_type in ["qdrant_local", "qdrant_cloud"]:
        location = ":memory:" if vectorstore_type == "qdrant_local" else kwargs.get("qdrant_url")
        return Qdrant.from_documents(
            docs,
            embeddings,
            location=location,
            collection_name=kwargs.get("qdrant_collection"),
            api_key=kwargs.get("qdrant_api_key")
        )
    
    else:
        raise ValueError(f"Unsupported vector store type: {vectorstore_type}")

# Main content
tab1, tab2 = st.tabs(["🎥 Video Analysis", "ℹ️ About"])

with tab1:
    # URL input
    url = st.text_input("Enter YouTube Video URL", 
                       placeholder="https://www.youtube.com/watch?v=...",
                       help="Paste the URL of the YouTube video you want to analyze")
    
    if url:
        try:
            # Extract video ID
            def extract_youtube_video_id(url):
                pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
                match = re.search(pattern, url)
                return match.group(1) if match else None
            
            video_id = extract_youtube_video_id(url)
            
            if video_id:
                # Display video thumbnail with improved handling
                try:
                    st.markdown(f"""
                        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                            <img src="https://img.youtube.com/vi/{video_id}/maxresdefault.jpg" 
                                 onerror="this.onerror=null;this.src='https://img.youtube.com/vi/{video_id}/0.jpg';" 
                                 class="thumbnail">
                        </div>
                    """, unsafe_allow_html=True)
                except:
                    st.markdown(f"""
                        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                            <img src="https://img.youtube.com/vi/{video_id}/0.jpg" class="thumbnail">
                        </div>
                    """, unsafe_allow_html=True)
                
                # Process transcript
                with st.spinner("Fetching and processing video transcript..."):
                    try:
                        # Use selected language
                        lang_code = language_options[selected_language]
                        languages = [lang_code] if lang_code else None
                        
                        transcript_list = YouTubeTranscriptApi.get_transcript(
                            video_id, 
                            languages=languages
                        )
                        transcript = " ".join(chunk["text"] for chunk in transcript_list)
                        
                        # Display transcript stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Transcript Length", f"{len(transcript.split())} words")
                        with col2:
                            st.metric("Chunks Created", len(transcript) // chunk_size + 1)
                        
                        # Split text
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap
                        )
                        chunks = splitter.create_documents([transcript])
                        
                        # Create vector store with selected embedding
                        with st.spinner(f"Loading {embedding_model_name} embeddings into {vectorstore_type}..."):
                            embeddings = HuggingFaceEmbeddings(
                                model_name=embedding_model_name,
                                model_kwargs={'device': 'cpu'}
                            )
                            
                            # Prepare connection parameters
                            connection_params = {}
                            if vectorstore_type == "chroma_persistent":
                                connection_params["persist_directory"] = persist_directory
                            elif vectorstore_type == "weaviate":
                                connection_params.update({
                                    "weaviate_url": weaviate_url,
                                    "weaviate_index": weaviate_index,
                                    "weaviate_api_key": weaviate_api_key or None
                                })
                            elif vectorstore_type == "pinecone":
                                connection_params.update({
                                    "pinecone_api_key": pinecone_api_key,
                                    "pinecone_env": pinecone_env,
                                    "pinecone_index": pinecone_index
                                })
                            elif vectorstore_type == "qdrant":
                                connection_params.update({
                                    "qdrant_url": qdrant_url,
                                    "qdrant_collection": qdrant_collection,
                                    "qdrant_api_key": qdrant_api_key or None
                                })
                            
                            # Create the appropriate vector store
                            vector_store = create_vector_store(
                                chunks,
                                embeddings,
                                vectorstore_type,
                                **connection_params
                            )
                            
                            if vector_store is None:
                                st.error("Failed to create vector store")
                                st.stop()
                            
                            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
                        
                        st.success("✅ Video transcript processed and stored successfully!")
                        
                        # Question input
                        question = st.text_input("Ask a question about the video", 
                                               placeholder="What is the main topic of this video?",
                                               key="question_input")
                        
                        if question and hf_token:
                            with st.spinner("Generating answer..."):
                                try:
                                    # Set up the LLM
                                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
                                    llm = HuggingFaceEndpoint(
                                        repo_id=llm_model_name,
                                        task="text-generation",
                                        temperature=temperature,
                                        max_new_tokens=max_length
                                    )
                                    chat_model = ChatHuggingFace(llm=llm)
                                    
                                    # Create the prompt template
                                    prompt = PromptTemplate(
                                        template="""
                                        You are a helpful assistant that answers questions about YouTube videos.
                                        Answer ONLY from the provided transcript context.
                                        If the context is insufficient, say you don't know.

                                        Context: {context}
                                        Question: {question}

                                        Answer:
                                        """,
                                        input_variables=['context', 'question']
                                    )
                                    
                                    # Create the RAG chain
                                    def format_docs(retrieved_docs):
                                        return "\n\n".join(doc.page_content for doc in retrieved_docs)
                                    
                                    chain = (
                                        RunnableParallel({
                                            "context": retriever | RunnableLambda(format_docs),
                                            "question": RunnablePassthrough()
                                        })
                                        | prompt
                                        | chat_model
                                        | StrOutputParser()
                                    )
                                    
                                    # Get the answer
                                    answer = chain.invoke(question)
                                    
                                    # Display answer
                                    st.subheader("Answer")
                                    st.markdown(f"""
                                        <div style="background-color: #e9ecef; padding: 15px; border-radius: 10px; margin-top: 10px;">
                                            {answer}
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show configuration used
                                    st.divider()
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.caption(f"**Generation model:** {llm_model_name}")
                                    with col2:
                                        st.caption(f"**Embedding model:** {embedding_model_name}")
                                    with col3:
                                        st.caption(f"**Vector store:** {vectorstore_type}")
                                    
                                except Exception as e:
                                    st.error(f"Error generating answer: {str(e)}")
                        
                    except TranscriptsDisabled:
                        st.error("No captions available for this video.")
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
            else:
                st.error("Invalid YouTube URL. Please check the URL and try again.")
        except Exception as e:
            st.error(f"Error extracting video ID: {str(e)}")

with tab2:
    st.header("About This App")
    st.markdown("""
        This application uses **Retrieval-Augmented Generation (RAG)** to answer questions about YouTube videos.
        
        ### How It Works:
        1. **Extracts** the transcript from a YouTube video
        2. **Processes** the text into chunks
        3. **Creates** embeddings using selected model
        4. **Stores** in selected vector database
        5. **Retrieves** relevant context for your question
        6. **Generates** an answer using selected LLM
        
        ### Vector Database Options:
        
        **Local Options:**
        - **FAISS**: Fast local vector store (no persistence)
        - **Chroma**: Local with optional persistence
        - **Weaviate**: Local or remote with advanced features
        - **Qdrant**: Local or cloud-based vector database
        
        **Cloud Options:**
        - **Pinecone**: Fully-managed vector database
        - **Qdrant Cloud**: Managed Qdrant service
        
        ### Model Selection:
        - Choose between **preset models** or enter your **custom models**
        - Configure generation, embedding, and vector store separately
        
        ### Technologies Used:
        - 🐍 Python
        - 🎥 YouTube Transcript API
        - 🔍 LangChain
        - 🤗 Hugging Face Models
        - 🎨 Streamlit
    """)
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6c757d; margin-top: 30px;">
            Made with ❤️ using Streamlit
        </div>
    """, unsafe_allow_html=True)