# RAG tools for LangGraph agent to query Superstore dataset
from langchain.tools import BaseTool
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field, PrivateAttr
from vector_store import SuperstoreVectorStore
from data_processor import SuperstoreDataProcessor
import json

class SuperstoreSearchInput(BaseModel):
    """Input schema for Superstore search tool"""
    query: str = Field(description="Search query to find relevant orders/products")
    limit: int = Field(default=5, description="Number of results to return (max 10)")
    category_filter: Optional[str] = Field(default=None, description="Filter by product category (Furniture, Office Supplies, Technology)")
    region_filter: Optional[str] = Field(default=None, description="Filter by region (West, East, Central, South)")

class SuperstoreStatsInput(BaseModel):
    """Input schema for Superstore statistics tool"""
    metric: str = Field(description="Type of statistics to retrieve: 'summary', 'categories', 'regions', 'top_customers', 'top_products'")

class SuperstoreSearchTool(BaseTool):
    """Tool for searching the Superstore dataset using vector similarity"""
    
    name: str = "superstore_search"
    description: str = """
    Search the Superstore dataset for relevant orders, products, or customer information.
    Use this tool when users ask about:
    - Specific products or orders
    - Customer information
    - Sales data
    - Product categories
    - Regional information
    
    The tool performs semantic search and can filter by category or region.
    """
    args_schema: Type[BaseModel] = SuperstoreSearchInput
    _vector_store: SuperstoreVectorStore = PrivateAttr()
    
    def __init__(self, vector_store: SuperstoreVectorStore):
        super().__init__()
        self._vector_store = vector_store
    
    def _run(self, query: str, limit: int = 5, category_filter: Optional[str] = None, region_filter: Optional[str] = None) -> str:
        """Execute the search"""
        try:
            # Limit results to max 10
            limit = min(limit, 10)
            
            # Build metadata filter
            where_filter = {}
            if category_filter:
                where_filter["category"] = {"$eq": category_filter}
            if region_filter:
                where_filter["region"] = {"$eq": region_filter}
            
            # Perform search
            results = self._vector_store.similarity_search(
                query=query,
                n_results=limit,
                where=where_filter if where_filter else None
            )
            
            if not results:
                return "No relevant results found for your query."
            
            # Format results for the LLM
            formatted_results = []
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                score = result["score"]
                
                formatted_result = f"""
Result {i} (Relevance: {score:.2f}):
- Order ID: {metadata.get('order_id', 'N/A')}
- Customer: {metadata.get('customer_name', 'N/A')}
- Product: {metadata.get('product_name', 'N/A')}
- Category: {metadata.get('category', 'N/A')} > {metadata.get('sub_category', 'N/A')}
- Location: {metadata.get('city', 'N/A')}, {metadata.get('state', 'N/A')} ({metadata.get('region', 'N/A')})
- Sales: ${metadata.get('sales', 0):.2f}
- Profit: ${metadata.get('profit', 0):.2f}
- Date: {metadata.get('order_date', 'N/A')}
"""
                formatted_results.append(formatted_result.strip())
            
            return f"Found {len(results)} relevant results:\n\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching dataset: {str(e)}"

class SuperstoreStatsTool(BaseTool):
    """Tool for getting statistics and insights from the Superstore dataset"""
    
    name: str = "superstore_stats"
    description: str = """
    Get statistical information and insights from the Superstore dataset.
    Use this tool when users ask about:
    - Overall sales performance
    - Dataset summary
    - Available categories or regions
    - Top performing customers or products
    
    Available metrics: 'summary', 'categories', 'regions', 'top_customers', 'top_products'
    """
    args_schema: Type[BaseModel] = SuperstoreStatsInput
    _data_processor: SuperstoreDataProcessor = PrivateAttr()
    
    def __init__(self, data_processor: SuperstoreDataProcessor):
        super().__init__()
        self._data_processor = data_processor
    
    def _run(self, metric: str) -> str:
        """Get statistics based on the requested metric"""
        try:
            if metric == "summary":
                stats = self._data_processor.get_summary_stats()
                return f"""
Dataset Summary:
- Total Records: {stats['total_records']:,}
- Total Sales: ${stats['total_sales']:,.2f}
- Total Profit: ${stats['total_profit']:,.2f}
- Unique Customers: {stats['unique_customers']:,}
- Unique Products: {stats['unique_products']:,}
- Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}
- Categories: {', '.join(stats['categories'])}
- Regions: {', '.join(stats['regions'])}
"""
            
            elif metric == "categories":
                if self._data_processor.df is None:
                    self._data_processor.load_data()
                
                category_stats = self._data_processor.df.groupby('Category').agg({
                    'Sales': ['sum', 'count'],
                    'Profit': 'sum'
                }).round(2)
                
                result = "Sales by Category:\n"
                for category in category_stats.index:
                    sales = category_stats.loc[category, ('Sales', 'sum')]
                    count = category_stats.loc[category, ('Sales', 'count')]
                    profit = category_stats.loc[category, ('Profit', 'sum')]
                    result += f"- {category}: ${sales:,.2f} sales, {count:,} orders, ${profit:,.2f} profit\n"
                
                return result
            
            elif metric == "regions":
                if self._data_processor.df is None:
                    self._data_processor.load_data()
                
                region_stats = self._data_processor.df.groupby('Region').agg({
                    'Sales': ['sum', 'count'],
                    'Profit': 'sum'
                }).round(2)
                
                result = "Sales by Region:\n"
                for region in region_stats.index:
                    sales = region_stats.loc[region, ('Sales', 'sum')]
                    count = region_stats.loc[region, ('Sales', 'count')]
                    profit = region_stats.loc[region, ('Profit', 'sum')]
                    result += f"- {region}: ${sales:,.2f} sales, {count:,} orders, ${profit:,.2f} profit\n"
                
                return result
            
            elif metric == "top_customers":
                if self._data_processor.df is None:
                    self._data_processor.load_data()
                
                top_customers = self._data_processor.df.groupby('Customer Name').agg({
                    'Sales': 'sum',
                    'Profit': 'sum',
                    'Order ID': 'nunique'
                }).sort_values('Sales', ascending=False).head(10)
                
                result = "Top 10 Customers by Sales:\n"
                for i, (customer, data) in enumerate(top_customers.iterrows(), 1):
                    result += f"{i}. {customer}: ${data['Sales']:,.2f} sales, {data['Order ID']} orders, ${data['Profit']:,.2f} profit\n"
                
                return result
            
            elif metric == "top_products":
                if self._data_processor.df is None:
                    self._data_processor.load_data()
                
                top_products = self._data_processor.df.groupby('Product Name').agg({
                    'Sales': 'sum',
                    'Profit': 'sum',
                    'Quantity': 'sum'
                }).sort_values('Sales', ascending=False).head(10)
                
                result = "Top 10 Products by Sales:\n"
                for i, (product, data) in enumerate(top_products.iterrows(), 1):
                    result += f"{i}. {product}: ${data['Sales']:,.2f} sales, {data['Quantity']} units, ${data['Profit']:,.2f} profit\n"
                
                return result
            
            else:
                return f"Unknown metric: {metric}. Available metrics: summary, categories, regions, top_customers, top_products"
                
        except Exception as e:
            return f"Error getting statistics: {str(e)}"

def create_rag_tools(vector_store: SuperstoreVectorStore, data_processor: SuperstoreDataProcessor) -> List[BaseTool]:
    """Create and return all RAG tools for the agent"""
    
    search_tool = SuperstoreSearchTool(vector_store=vector_store)
    stats_tool = SuperstoreStatsTool(data_processor=data_processor)
    
    return [search_tool, stats_tool]