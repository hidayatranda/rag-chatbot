# Nxtneura RAG Chatbot 
url: https://nxtneura.streamlit.app/

Chatbot berbasis RAG (Retrieval-Augmented Generation) yang dibangun menggunakan Streamlit, LangGraph, dan Google Gemini untuk analisis dataset Superstore e-commerce.

## Fitur Utama

- **RAG Implementation**: Menggunakan ChromaDB untuk vector storage dan semantic search
- **LangGraph Agent**: Agent berbasis ReAct pattern untuk reasoning dan tool usage
- **Google Gemini Integration**: Menggunakan model Gemini 2.5 Flash untuk natural language processing
- **Interactive UI**: Interface web yang user-friendly dengan Streamlit
- **Dataset Analysis**: Analisis mendalam dataset Superstore
- **Real-time Search**: Pencarian semantik real-time pada data penjualan

## Struktur Proyek

```
RAG CHATBOT/
├── streamlit_rag_app.py      # Aplikasi utama Streamlit
├── data_processor.py         # Pemrosesan dan transformasi data
├── vector_store.py          # Implementasi ChromaDB vector store
├── rag_tools.py            # Tools untuk LangGraph agent
├── requirements.txt        # Dependencies Python
├── Superstore Dataset - Orders.csv  # Dataset e-commerce
├── chroma_db/             # Database vector ChromaDB
└── README.md             # Dokumentasi proyek
```

## Komponen Teknis

### 1. Data Processor (`data_processor.py`)
- **Class**: `SuperstoreDataProcessor`
- **Fungsi**: Memuat dan memproses dataset CSV Superstore
- **Features**:
  - Loading data dari CSV dengan error handling
  - Konversi rows menjadi documents untuk vector storage
  - Statistik summary (total sales, profit, customers)
  - Search dan filtering berdasarkan kriteria

### 2. Vector Store (`vector_store.py`)
- **Class**: `SuperstoreVectorStore`
- **Database**: ChromaDB dengan persistent storage
- **Embedding Model**: `all-MiniLM-L6-v2` dari SentenceTransformers
- **Features**:
  - Semantic similarity search
  - Metadata filtering (category, region)
  - Collection statistics
  - Automatic document indexing

### 3. RAG Tools (`rag_tools.py`)
- **SuperstoreSearchTool**: Pencarian semantik pada dataset
- **SuperstoreStatsTool**: Statistik dan insights dataset
- **Integration**: Tools terintegrasi dengan LangGraph agent
- **Input Validation**: Menggunakan Pydantic schemas

### 4. Main Application (`streamlit_rag_app.py`)
- **Framework**: Streamlit untuk web interface
- **Agent**: LangGraph ReAct agent dengan Google Gemini
- **Session Management**: Streamlit session state untuk conversation history
- **UI Components**: Sidebar settings, chat interface, suggestion buttons

## Instalasi dan Setup

### Prerequisites
- Python 3.8+
- Google AI API Key (untuk Gemini)

### Langkah Instalasi

1. **Clone repository**
```bash
git clone <repository-url>
cd "RAG CHATBOT"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup environment**
- Pastikan file `Superstore Dataset - Orders.csv` ada di root directory
- Siapkan Google AI API Key

4. **Jalankan aplikasi**
```bash
streamlit run streamlit_rag_app.py
```

5. **Akses aplikasi**
- Buka browser dan akses `http://localhost:8501`
- Masukkan Google AI API Key di sidebar
- Mulai chat dengan bot

## Dataset Information

**Superstore Dataset - Orders.csv**
- **Total Records**: 9,994 orders
- **Columns**: 21 kolom (Order ID, Customer info, Product details, Sales metrics)
- **Categories**: Furniture, Office Supplies, Technology
- **Regions**: West, East, Central, South
- **Time Range**: 2014-2017
- **Metrics**: Sales, Profit, Quantity, Discount

## Capabilities Chatbot

### 1. Search & Query
- Pencarian produk berdasarkan nama atau kategori
- Informasi customer dan order history
- Filter berdasarkan region atau kategori

### 2. Analytics & Statistics
- Summary statistik penjualan
- Top customers dan products
- Performance analysis by category/region
- Trend analysis dan insights

### 3. Natural Language Processing
- Pemahaman query dalam bahasa natural
- Contextual responses berdasarkan data
- Multi-turn conversations dengan memory

## Contoh Usage

### Query Examples:
```
"Siapa customer dengan pembelian terbesar?"
"Produk apa yang paling menguntungkan?"
"Berapa total penjualan di region West?"
"Tampilkan statistik kategori Technology"
"Cari order dari customer John Smith"
```

### Response Features:
- Data-driven answers dari dataset
- Formatted output dengan metrics
- Contextual follow-up suggestions
- Error handling untuk invalid queries

## Performance & Optimization

### Caching Strategy
- `@st.cache_resource` untuk RAG components
- Persistent ChromaDB storage
- Session state management untuk conversation history

### Vector Search Optimization
- Efficient embedding dengan SentenceTransformers
- Metadata filtering untuk faster queries
- Configurable result limits (max 10)

### Memory Management
- Lazy loading untuk large datasets
- Efficient document chunking
- Optimized vector storage format

## Security & Best Practices

- API key input dengan password masking
- Error handling untuk semua components
- Input validation dengan Pydantic
- Safe file operations dengan path checking

## Future Enhancements

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Export functionality untuk results
- [ ] Custom embedding models
- [ ] Real-time data updates
- [ ] User authentication system

## Dependencies

```
streamlit              # Web framework
matplotlib            # Data visualization
google-genai>=1.0.0   # Google Gemini API
langchain-google-genai>=2.1.0  # LangChain Gemini integration
langgraph>=0.0.30     # Agent framework
langchain>=0.1.0      # LangChain core
chromadb>=0.4.0       # Vector database
pandas>=2.0.0         # Data manipulation
sentence-transformers>=2.2.0  # Embedding models
langchain-community>=0.0.20   # LangChain community tools
langchain-chroma>=0.1.0       # ChromaDB integration
```

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License.

## Support

Untuk pertanyaan atau issues, silakan buat issue di repository atau hubungi tim development.

---

**Nxtneura RAG Chatbot** - Intelligent e-commerce data analysis with conversational AI
