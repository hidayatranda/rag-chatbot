# Data processing module for Superstore dataset
import pandas as pd
import os
from typing import List, Dict, Any

class SuperstoreDataProcessor:
    """
    Class to handle loading and processing of Superstore dataset for RAG implementation
    """
    
    def __init__(self, csv_path: str = "Superstore Dataset - Orders.csv"):
        self.csv_path = csv_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV data into a pandas DataFrame"""
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")
            
            self.df = pd.read_csv(self.csv_path)
            print(f"Successfully loaded {len(self.df)} records from {self.csv_path}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_documents(self) -> List[Dict[str, Any]]:
        """
        Convert DataFrame rows into documents suitable for vector storage
        Each row becomes a document with metadata
        """
        if self.df is None:
            self.load_data()
        
        documents = []
        
        for idx, row in self.df.iterrows():
            # Create a comprehensive text representation of each order
            document_text = f"""
            Order Information:
            - Order ID: {row['Order ID']}
            - Order Date: {row['Order Date']}
            - Ship Date: {row['Ship Date']}
            - Ship Mode: {row['Ship Mode']}
            
            Customer Information:
            - Customer ID: {row['Customer ID']}
            - Customer Name: {row['Customer Name']}
            - Segment: {row['Segment']}
            - Location: {row['City']}, {row['State']}, {row['Country']}
            - Region: {row['Region']}
            
            Product Information:
            - Product ID: {row['Product ID']}
            - Category: {row['Category']}
            - Sub-Category: {row['Sub-Category']}
            - Product Name: {row['Product Name']}
            
            Financial Information:
            - Sales: ${row['Sales']:.2f}
            - Quantity: {row['Quantity']}
            - Discount: {row['Discount']:.2%}
            - Profit: ${row['Profit']:.2f}
            """
            
            # Create metadata for filtering and additional context
            metadata = {
                "row_id": int(row['Row ID']),
                "order_id": str(row['Order ID']),
                "customer_name": str(row['Customer Name']),
                "category": str(row['Category']),
                "sub_category": str(row['Sub-Category']),
                "product_name": str(row['Product Name']),
                "city": str(row['City']),
                "state": str(row['State']),
                "region": str(row['Region']),
                "segment": str(row['Segment']),
                "sales": float(row['Sales']),
                "profit": float(row['Profit']),
                "order_date": str(row['Order Date'])
            }
            
            documents.append({
                "content": document_text.strip(),
                "metadata": metadata
            })
        
        print(f"Prepared {len(documents)} documents for vector storage")
        return documents
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset"""
        if self.df is None:
            self.load_data()
        
        stats = {
            "total_records": len(self.df),
            "total_sales": self.df['Sales'].sum(),
            "total_profit": self.df['Profit'].sum(),
            "unique_customers": self.df['Customer ID'].nunique(),
            "unique_products": self.df['Product ID'].nunique(),
            "categories": self.df['Category'].unique().tolist(),
            "regions": self.df['Region'].unique().tolist(),
            "date_range": {
                "start": self.df['Order Date'].min(),
                "end": self.df['Order Date'].max()
            }
        }
        
        return stats
    
    def search_by_criteria(self, **criteria) -> pd.DataFrame:
        """
        Search the dataset by various criteria
        Example: search_by_criteria(category='Furniture', region='West')
        """
        if self.df is None:
            self.load_data()
        
        filtered_df = self.df.copy()
        
        for key, value in criteria.items():
            if key in filtered_df.columns:
                if isinstance(value, str):
                    filtered_df = filtered_df[filtered_df[key].str.contains(value, case=False, na=False)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]
        
        return filtered_df