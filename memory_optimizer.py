# Memory optimization utilities for Windows systems
import gc
import os
import psutil
import warnings

def optimize_memory_usage():
    """
    Optimize memory usage for the RAG application on Windows
    """
    # Suppress unnecessary warnings to reduce memory overhead
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Force garbage collection
    gc.collect()
    
    # Set environment variables for memory optimization
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Reduce tokenizer memory usage
    os.environ['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '2'  # Limit MKL threads
    
    print("Memory optimization settings applied")

def check_memory_usage():
    """
    Check current memory usage and provide recommendations
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.1f} MB ({memory_percent:.1f}%)")
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / 1024 / 1024 / 1024
        
        print(f"Available system memory: {available_gb:.1f} GB")
        
        if available_gb < 2:
            print("WARNING: Low available memory. Consider closing other applications.")
            
        return memory_info.rss / 1024 / 1024  # Return MB
        
    except Exception as e:
        print(f"Could not check memory usage: {e}")
        return None

def force_cleanup():
    """
    Force memory cleanup
    """
    gc.collect()
    print("Memory cleanup completed")