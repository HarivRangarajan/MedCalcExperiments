"""API utilities for OpenAI and other services."""

import os
from typing import Optional


def load_api_key() -> Optional[str]:
    """Load OpenAI API key from environment or local config.
    
    Returns:
        str: The API key if found and valid, None otherwise
    """
    # Try environment variable first
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != "your-api-key-here":
        print("✅ API key loaded from environment variable")
        return api_key
    
    # Try local config file
    try:
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key-here":
            print("✅ API key loaded from local config.py")
            return OPENAI_API_KEY
    except ImportError:
        print("⚠️  Local config.py not found")
        pass
    
    print("❌ No valid API key found")
    return None


def create_openai_client(api_key: str):
    """Create an OpenAI client instance.
    
    Args:
        api_key: The OpenAI API key
        
    Returns:
        OpenAI client instance
        
    Raises:
        ImportError: If openai library is not installed
    """
    from openai import OpenAI
    return OpenAI(api_key=api_key)

