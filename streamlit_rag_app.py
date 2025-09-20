# Import the necessary libraries
import streamlit as st  # For creating the web app interface
from langchain_google_genai import ChatGoogleGenerativeAI  # For interacting with Google Gemini via LangChain
from langgraph.prebuilt import create_react_agent  # For creating a ReAct agent
from langchain_core.messages import HumanMessage, AIMessage  # For message formatting
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import matplotlib.pyplot as plt
import numpy as np
import os
from vector_store import SuperstoreVectorStore, initialize_vector_store
from data_processor import SuperstoreDataProcessor
from rag_tools import create_rag_tools
from memory_optimizer import optimize_memory_usage, check_memory_usage, force_cleanup

# --- 1. Page Configuration and Title ---

# Set the title and a caption for the web page
st.title("Nxtneura Chatbot")
st.caption("A simple and friendly chat using RAG with Google's Gemini model")

# --- 2. Sidebar for Settings ---

# Create a sidebar section for app settings using 'with st.sidebar:'
with st.sidebar:
    # Add a subheader to organize the settings
    st.subheader("Settings")
    
    # Create a text input field for the Google AI API Key.
    # 'type="password"' hides the key as the user types it.
    google_api_key = st.text_input("Google AI API Key", type="password")
    
    # Create a button to reset the conversation.
    # 'help' provides a tooltip that appears when hovering over the button.
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

# --- 3. API Key and Agent Initialization ---

# Check if the user has provided an API key.
# If not, display an informational message and stop the app from running further.
if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.")
    st.stop()

# --- 3. Initialize RAG Components ---
@st.cache_resource
def initialize_rag_components():
    """Initialize all RAG components with error handling"""
    
    try:
        # Memory optimization
        optimize_memory_usage()
        force_cleanup()
        
        # Initialize data processor
        data_processor = SuperstoreDataProcessor()
        force_cleanup()
        
        # Initialize vector store
        vector_store = initialize_vector_store(data_processor)
        force_cleanup()
        
        # Create RAG tools
        tools = create_rag_tools(vector_store, data_processor)
        force_cleanup()
        
        return data_processor, vector_store, tools
        
    except Exception as e:
        st.error(f"Error initializing RAG components: {str(e)}")
        st.warning("Trying fallback mode with data processor only...")
        
        try:
            # Fallback: only initialize data processor
            data_processor = SuperstoreDataProcessor()
            data_processor.load_data()
            
            return data_processor, None, None
        except Exception as fallback_error:
            st.error(f"Fallback also failed: {str(fallback_error)}")
            force_cleanup()
            return None, None, None

# Initialize RAG components
data_processor, vector_store, tools = initialize_rag_components()

# This block of code handles the creation of the LangGraph agent.
# It's designed to be efficient: it only creates a new agent if one doesn't exist
# or if the user has changed the API key in the sidebar.

# We use `st.session_state` which is Streamlit's way of "remembering" variables
# between user interactions (like sending a message or clicking a button).
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        # Initialize the LLM with the API key
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

        # Create the agent with RAG tools
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=tools if tools else [],  # Use RAG tools if available
            prompt="""You are a helpful assistant with access to the Superstore dataset. 
            You can help users analyze sales data, find specific orders, get statistics, and answer questions about customers, products, and business performance.
            
            When users ask about the dataset, use the available tools to search for relevant information or get statistics.
            Always provide clear, helpful responses based on the data you find.
            
            Available capabilities:
            - Search for specific orders, products, or customers
            - Get summary statistics and insights
            - Analyze sales performance by category, region, or time period
            - Find top customers and products
            
            Be conversational and helpful in your responses.""",
            debug=True
        )
        
        # Store the new key in session state to compare against later.
        st.session_state._last_key = google_api_key
        # Since the key changed, we must clear the old message history.
        st.session_state.pop("messages", None)
    except Exception as e:
        # If the key is invalid, show an error and stop.
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

# --- 4. Chat History Management ---

# Initialize the message history (as a list) if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle the reset button click.
if reset_button:
    # If the reset button is clicked, clear the agent and message history from memory.
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    # st.rerun() tells Streamlit to refresh the page from the top.
    st.rerun()

# --- 5. Sidebar with Dataset Information ---
if data_processor and vector_store:
    with st.sidebar:
        st.header("Dataset Information")
        
        # Get basic stats
        try:
            stats = data_processor.get_summary_stats()
            st.metric("Total Records", f"{stats['total_records']:,}")
            st.metric("Total Sales", f"${stats['total_sales']:,.2f}")
            st.metric("Total Profit", f"${stats['total_profit']:,.2f}")
            st.metric("Unique Customers", f"{stats['unique_customers']:,}")
            
            st.subheader("Available Categories")
            for category in stats['categories']:
                st.write(f"• {category}")
            
            st.subheader("Available Regions")
            for region in stats['regions']:
                st.write(f"• {region}")
                
            # Vector store info
            collection_stats = vector_store.get_collection_stats()
            st.subheader("Vector Database")
            st.write(f"Documents indexed: {collection_stats['total_documents']:,}")
            
        except Exception as e:
            st.error(f"Error loading dataset info: {str(e)}")

# --- 6. Suggestion Questions ---

# Display suggested questions (always visible)
st.markdown("**Pilih salah satu pertanyaan berikut untuk memulai analisis:**")

# Add custom CSS for compact suggestion buttons
st.markdown("""
<style>
/* Reset and base styles */
.suggestion-container {
    margin: 15px 0;
    clear: both;
    width: 100%;
}

/* Button styling */
.stButton > button {
    width: 100% !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    background: #2d2d2d !important;
    border: 1px solid #404040 !important;
    border-radius: 6px !important;
    color: #cccccc !important;
    font-weight: 400 !important;
    font-size: 13px !important;
    text-align: center !important;
    padding: 0 8px !important;
    margin: 0 0 8px 0 !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    position: relative !important;
    overflow: hidden !important;
    line-height: 1.2 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
    box-sizing: border-box !important;
}

.stButton > button:hover {
    background: #3a3a3a !important;
    border: 1px solid #555555 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}

.stButton > button:active {
    background: #404040 !important;
    transform: translateY(0px) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}

/* Column layout fixes */
.stColumn {
    padding: 0 2px !important;
    margin: 0 !important;
    min-width: 0 !important;
    flex: 1 !important;
}

.stColumn > div {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Container for suggestion buttons */
.suggestion-buttons {
    display: block !important;
    margin: 20px 0 !important;
    width: 100% !important;
    clear: both !important;
    overflow: hidden !important;
}

.suggestion-row {
    display: flex !important;
    gap: 4px !important;
    margin-bottom: 8px !important;
    width: 100% !important;
    align-items: stretch !important;
}

/* Ensure buttons in columns don't overlap */
.suggestion-row .stColumn .stButton {
    margin: 0 !important;
    width: 100% !important;
    display: block !important;
}

.suggestion-row .stColumn .stButton > button {
    margin: 0 !important;
    width: 100% !important;
}

/* Chat input styling */
.stChatInput > div > div > input {
    background-color: #1e1e1e !important;
    border: 1px solid #404040 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}

.stChatInput > div > div > input::placeholder {
    color: #888888 !important;
    opacity: 1 !important;
    font-style: italic !important;
}

.stChatInput > div > div > input:focus {
    border-color: #0066cc !important;
    box-shadow: 0 0 0 1px #0066cc !important;
    outline: none !important;
}
</style>""", unsafe_allow_html=True)

# --- 7. Display Past Messages ---

# Loop through every message currently stored in the session state.
for msg in st.session_state.messages:
    # For each message, create a chat message bubble with the appropriate role ("user" or "assistant").
    with st.chat_message(msg["role"]):
        # Display the content of the message using Markdown for nice formatting.
        st.markdown(msg["content"])

# --- 8. Chat Input and Response Generation ---

def handle_direct_query(data_processor, query):
    """Handle queries directly using data processor when RAG tools fail"""
    if data_processor is None or data_processor.df is None:
        return "Data tidak tersedia untuk query ini."
    
    df = data_processor.df
    query_lower = query.lower()
    
    try:
        # Handle top customers queries
        if any(keyword in query_lower for keyword in ['pelanggan teratas', 'top customer', 'customer teratas', 'pelanggan terbaik']):
            top_customers = df.groupby('Customer Name').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Order ID': 'nunique'
            }).sort_values('Sales', ascending=False).head(5)
            
            result = "🏆 **Top 5 Pelanggan Berdasarkan Penjualan:**\n\n"
            for i, (customer, data) in enumerate(top_customers.iterrows(), 1):
                result += f"{i}. **{customer}**\n"
                result += f"   - Sales: ${data['Sales']:,.2f}\n"
                result += f"   - Orders: {data['Order ID']}\n"
                result += f"   - Profit: ${data['Profit']:,.2f}\n\n"
            return result
        
        # Handle furniture profit queries
        elif any(keyword in query_lower for keyword in ['furniture', 'profit tertinggi', 'produk furniture']):
            furniture_df = df[df['Category'] == 'Furniture']
            if len(furniture_df) == 0:
                return "Tidak ada produk furniture ditemukan dalam dataset."
            
            top_furniture = furniture_df.groupby('Product Name').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Quantity': 'sum'
            }).sort_values('Profit', ascending=False).head(5)
            
            result = "🪑 **Top 5 Produk Furniture dengan Profit Tertinggi:**\n\n"
            for i, (product, data) in enumerate(top_furniture.iterrows(), 1):
                result += f"{i}. **{product}**\n"
                result += f"   - Sales: ${data['Sales']:,.2f}\n"
                result += f"   - Profit: ${data['Profit']:,.2f}\n"
                result += f"   - Quantity: {data['Quantity']}\n\n"
            return result
        
        # Handle general statistics queries
        elif any(keyword in query_lower for keyword in ['statistik', 'summary', 'ringkasan', 'total']):
            stats = data_processor.get_summary_stats()
            result = "**Ringkasan Dataset Superstore:**\n\n"
            result += f"- **Total Records:** {stats['total_records']:,}\n"
            result += f"- **Total Sales:** ${stats['total_sales']:,.2f}\n"
            result += f"- **Total Profit:** ${stats['total_profit']:,.2f}\n"
            result += f"- **Unique Customers:** {stats['unique_customers']:,}\n"
            result += f"- **Unique Products:** {stats['unique_products']:,}\n"
            result += f"- **Categories:** {', '.join(stats['categories'])}\n"
            result += f"- **Regions:** {', '.join(stats['regions'])}\n"
            return result
        
        else:
            return "Query tidak dikenali. Coba tanyakan tentang:\n- Pelanggan teratas\n- Produk furniture dengan profit tertinggi\n- Statistik dataset"
            
    except Exception as e:
        return f"Error memproses query: {str(e)}"

# Handle suggested question clicks
if "suggested_question" in st.session_state:
    user_input = st.session_state.suggested_question
    # Remove the suggested question from session state to prevent re-processing
    del st.session_state.suggested_question
else:
    user_input = None

# Create a two-row layout of compact suggestion buttons right above chat input
st.markdown('<div class="suggestion-buttons">', unsafe_allow_html=True)

# First row - 4 buttons
st.markdown('<div class="suggestion-row">', unsafe_allow_html=True)
cols1 = st.columns(4)

with cols1[0]:
    if st.button("Statistik penjualan", key="q1"):
        st.session_state.suggested_question = "Tampilkan statistik penjualan keseluruhan"
        st.rerun()

with cols1[1]:
    if st.button("Top 5 pelanggan", key="q2"):
        st.session_state.suggested_question = "Siapa 5 pelanggan teratas berdasarkan penjualan?"
        st.rerun()

with cols1[2]:
    if st.button("Performa region", key="q3"):
        st.session_state.suggested_question = "Bagaimana performa penjualan di setiap region?"
        st.rerun()

with cols1[3]:
    if st.button("Furniture profit", key="q4"):
        st.session_state.suggested_question = "Cari produk furniture dengan profit tertinggi"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Second row - 4 buttons
st.markdown('<div class="suggestion-row">', unsafe_allow_html=True)
cols2 = st.columns(4)

with cols2[0]:
    if st.button("Kategori produk", key="q5"):
        st.session_state.suggested_question = "Tampilkan kategori produk yang tersedia"
        st.rerun()

with cols2[1]:
    if st.button("Penjualan tertinggi", key="q7"):
        st.session_state.suggested_question = "Cari pesanan dengan nilai penjualan tertinggi"
        st.rerun()

with cols2[2]:
    if st.button("Segmen aktif", key="q8"):
        st.session_state.suggested_question = "Analisis segmen pelanggan mana yang paling aktif"
        st.rerun()

with cols2[3]:
    if st.button("Produk profit", key="q6"):
        st.session_state.suggested_question = "Produk apa yang paling menguntungkan?"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Always show the chat input, regardless of whether there are messages or not
chat_input = st.chat_input("Tanyakan tentang sales ecommerce anda !")

# Use chat input if no suggested question was clicked
if user_input is None:
    user_input = chat_input

# If the user has entered a message, process it
if user_input:
    # Add the user's message to the session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display the user's message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Searching dataset and generating response..."):
            try:
                # Try RAG tools first if available
                if tools is not None:
                    try:
                        # Get response from the agent
                        response = st.session_state.agent.invoke({"messages": [HumanMessage(content=user_input)]})
                        
                        # Extract the assistant's message
                        assistant_message = response["messages"][-1].content
                        
                    except Exception as rag_error:
                        st.warning(f"RAG tools error: {str(rag_error)}")
                        st.info("Menggunakan fallback mode...")
                        assistant_message = handle_direct_query(data_processor, user_input)
                
                else:
                    # Use direct query handling when RAG tools not available
                    st.info("Menggunakan direct data access...")
                    assistant_message = handle_direct_query(data_processor, user_input)
                
                # Display the response
                st.write(assistant_message)
                
                # Add the assistant's response to the session state
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})