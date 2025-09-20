# Vector store implementation using ChromaDB
# Updated for ChromaDB v1.0+ compatibility
import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
import uuid
from data_processor import SuperstoreDataProcessor

class SuperstoreVectorStore:
    """
    Vector store implementation for Superstore dataset using ChromaDB
    """
    
    def __init__(self, 
                 collection_name: str = "superstore_orders",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model with memory optimization
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device='cpu'  # Force CPU usage to reduce memory pressure
        )
        
        # Initialize ChromaDB client with robust error handling
        try:
            # First try EphemeralClient to test basic functionality
            print("Testing ChromaDB connection...")
            test_client = chromadb.EphemeralClient()
            test_client.heartbeat()
            print("ChromaDB basic connection test passed")
            
            # Use the original persist directory
            print(f"Initializing PersistentClient at: {persist_directory}")
            self.client = chromadb.PersistentClient(path=persist_directory)
            print("ChromaDB PersistentClient initialized successfully")
            
        except Exception as e:
            print(f"Error with PersistentClient: {e}")
            print("Falling back to EphemeralClient...")
            self.client = chromadb.EphemeralClient()
            self.persist_directory = None
            print("Using EphemeralClient (data will not persist between sessions)")
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            print(f"Found existing collection: {self.collection_name}")
            return collection
        except:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Superstore orders dataset for RAG"}
            )
            print(f"Created new collection: {self.collection_name}")
            return collection
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for documents with memory optimization"""
        texts = [doc["content"] for doc in documents]
        
        # Process embeddings in smaller batches to reduce memory usage
        batch_size = 50  # Reduced from processing all at once
        all_embeddings = []
        
        print(f"Generating embeddings for {len(texts)} documents in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            
            # Generate embeddings for this batch
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                show_progress_bar=True,
                batch_size=16  # Further reduce internal batch size
            )
            
            all_embeddings.extend(batch_embeddings.tolist())
            print(f"Processed embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return all_embeddings
    
    def add_documents(self, documents: List[Dict[str, Any]], force_reload: bool = False):
        """Add documents to the vector store"""
        
        # Check if collection already has documents
        if self.collection.count() > 0 and not force_reload:
            print(f"Collection already contains {self.collection.count()} documents. Use force_reload=True to reload.")
            return
        
        if force_reload and self.collection.count() > 0:
            print("Force reloading: Clearing existing collection...")
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Generate embeddings
        embeddings = self.embed_documents(documents)
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add to collection in batches
        batch_size = 50  # Reduced batch size for memory optimization
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            # Force garbage collection after each batch to free memory
            import gc
            gc.collect()
        
        print(f"Successfully added {len(documents)} documents to vector store")
    
    def similarity_search(self, 
                         query: str, 
                         n_results: int = 5,
                         where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector store
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter
        
        Returns:
            List of matching documents with metadata and scores
        """
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity score
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        if count == 0:
            return {"total_documents": 0, "message": "Collection is empty"}
        
        # Get a sample of documents to analyze
        sample = self.collection.get(limit=min(100, count), include=["metadatas"])
        
        categories = set()
        regions = set()
        
        for metadata in sample["metadatas"]:
            if "category" in metadata:
                categories.add(metadata["category"])
            if "region" in metadata:
                regions.add(metadata["region"])
        
        return {
            "total_documents": count,
            "sample_categories": list(categories),
            "sample_regions": list(regions),
            "embedding_model": self.embedding_model_name
        }
    
    def initialize_with_data(self, csv_path: str = "Superstore Dataset - Orders.csv", force_reload: bool = False):
        """Initialize the vector store with Superstore data"""
        
        # Load and process data
        processor = SuperstoreDataProcessor(csv_path)
        documents = processor.prepare_documents()
        
        # Add to vector store
        self.add_documents(documents, force_reload=force_reload)
        
        return self.get_collection_stats()

# Utility function to initialize vector store
def setup_vector_store(force_reload: bool = False) -> SuperstoreVectorStore:
    """Setup and initialize the vector store with Superstore data"""
    
    vector_store = SuperstoreVectorStore()
    stats = vector_store.initialize_with_data(force_reload=force_reload)
    
    print("Vector store setup complete!")
    print(f"Statistics: {stats}")
    
    return vector_store

def initialize_vector_store(data_processor: SuperstoreDataProcessor, force_reload: bool = False) -> SuperstoreVectorStore:
    """Initialize vector store with provided data processor"""
    
    vector_store = SuperstoreVectorStore()
    
    # Get documents from the data processor
    documents = data_processor.prepare_documents()
    
    # Add documents to vector store
    vector_store.add_documents(documents, force_reload=force_reload)
    
    print("Vector store initialized with data processor!")
    stats = vector_store.get_collection_stats()
    print(f"Statistics: {stats}")
    
    return vector_store