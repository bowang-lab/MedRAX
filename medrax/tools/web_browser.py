"""Web browser tool for MedRAX2.

This module implements a web browsing tool for MedRAX2, allowing the agent
to search the web, visit URLs, and extract information from web pages.
"""

import os
import re
import json
import time
from typing import Dict, Optional, Any, Type, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class WebBrowserSchema(BaseModel):
    """Schema for web browser tool."""
    query: str = Field("", description="The search query (leave empty if visiting a URL)")
    url: str = Field("", description="The URL to visit (leave empty if performing a search)")


class SearchQuerySchema(BaseModel):
    """Schema for web search queries."""
    query: str = Field(..., description="The search query string")


class VisitUrlSchema(BaseModel):
    """Schema for URL visits."""
    url: str = Field(..., description="The URL to visit")


class WebBrowserTool(BaseTool):
    """Tool for browsing the web, searching for information, and visiting URLs.
    
    This tool provides the agent with internet browsing capabilities, including:
    1. Performing web searches using a search engine API
    2. Visiting specific URLs and extracting their content
    3. Following links within pages
    """
    name: str = "WebBrowserTool"
    description: str = "Search the web for information or visit specific URLs to retrieve content"
    search_api_key: Optional[str] = None
    search_engine_id: Optional[str] = None
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    max_results: int = 5
    args_schema: Type[BaseModel] = WebBrowserSchema
    
    def __init__(self, search_api_key: Optional[str] = None, search_engine_id: Optional[str] = None, **kwargs):
        """Initialize the web browser tool.
        
        Args:
            search_api_key: Google Custom Search API key (optional)
            search_engine_id: Google Custom Search Engine ID (optional)
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        # Try to get API keys from environment variables if not provided
        self.search_api_key = search_api_key or os.environ.get("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = search_engine_id or os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
        
    def search_web(self, query: str) -> Dict[str, Any]:
        """Search the web using Google Custom Search API.
        
        Args:
            query: The search query string
            
        Returns:
            Dict containing search results
        """
        if not self.search_api_key or not self.search_engine_id:
            return {
                "error": "Search API key or engine ID not configured. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables."
            }
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.search_api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": self.max_results
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            if "items" not in results:
                return {"results": [], "message": "No results found"}
                
            formatted_results = []
            for item in results["items"]:
                formatted_results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "source": item.get("displayLink")
                })
                
            return {
                "results": formatted_results,
                "message": f"Found {len(formatted_results)} results for query: {query}"
            }
            
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    def visit_url(self, url: str) -> Dict[str, Any]:
        """Visit a URL and extract its content.
        
        Args:
            url: The URL to visit
            
        Returns:
            Dict containing the page content, title, and metadata
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {"error": f"Invalid URL: {url}"}
                
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            
            # Extract main content (remove scripts, styles, etc.)
            for script in soup(["script", "style", "meta", "noscript"]):
                script.extract()
                
            # Get text content
            text_content = soup.get_text(separator="\n", strip=True)
            # Clean up whitespace
            text_content = re.sub(r'\n+', '\n', text_content)
            text_content = re.sub(r' +', ' ', text_content)
            
            # Extract links
            links = []
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # Handle relative URLs
                if href.startswith("/"):
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    href = base_url + href
                if href.startswith(("http://", "https://")):
                    links.append({
                        "text": link.get_text(strip=True) or href,
                        "url": href
                    })
            
            # Extract images (limited to first 3)
            images = []
            for i, img in enumerate(soup.find_all("img", src=True)[:3]):
                src = img["src"]
                # Handle relative URLs
                if src.startswith("/"):
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    src = base_url + src
                if src.startswith(("http://", "https://")):
                    images.append(src)
            
            return {
                "title": title,
                "content": text_content[:10000] if len(text_content) > 10000 else text_content,
                "url": url,
                "links": links[:10],  # Limit to 10 links
                "images": images,
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": len(text_content),
                "truncated": len(text_content) > 10000
            }
            
        except Exception as e:
            return {"error": f"Failed to visit {url}: {str(e)}"}

    async def _arun(self, query: str = "", url: str = "") -> str:
        """Run the tool asynchronously."""
        return json.dumps(self._run(query=query, url=url))
    
    def _run(self, query: str = "", url: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the web browser tool.
    
        Args:
            query: Search query (if searching)
            url: URL to visit (if visiting a specific page)
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the results and metadata
        """
        metadata = {
            "query": query if query else "",
            "url": url if url else "",
            "timestamp": time.time(),
            "tool": "WebBrowserTool"
        }
        
        if url:
            result = self.visit_url(url)
            return result, metadata
        elif query:
            result = self.search_web(query)
            return result, metadata
        else:
            return {"error": "Please provide either a search query or a URL to visit"}, metadata

