"""Web browser tool for MedRAX2.

This module implements a web browsing tool for MedRAX2, allowing the agent
to search the web, visit URLs, and extract information from web pages.
"""

import os
import re
import time
from typing import Dict, Optional, Any, Type, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class WebBrowserSchema(BaseModel):
    """Schema for web browser tool input."""

    query: str = Field("", description="The search query (leave empty if visiting a URL)")
    url: str = Field("", description="The URL to visit (leave empty if performing a search)")
    max_content_length: int = Field(
        5000, description="Maximum length of text content to extract (default: 5000 characters)"
    )
    max_links: int = Field(5, description="Maximum number of links to extract (default: 5)")


class SearchQuerySchema(BaseModel):
    """Schema for web search queries."""

    query: str = Field(..., description="The search query string")


class VisitUrlSchema(BaseModel):
    """Schema for URL visits."""

    url: str = Field(..., description="The URL to visit")


class WebBrowserTool(BaseTool):
    """Tool for browsing the web and retrieving information from online sources.

    This tool provides comprehensive internet browsing capabilities for the medical agent,
    enabling access to current medical information, research papers, clinical guidelines,
    and other online resources. It supports both web search functionality and direct URL access.

    Key capabilities:
    - Web search using Google Custom Search API for targeted information retrieval
    - Direct URL access for visiting specific medical websites and resources
    - Content extraction and parsing from web pages with structured output
    - Link extraction for discovering related resources (configurable limit)
    - Image detection and metadata extraction from medical websites
    - Configurable content length limits for efficient processing
    - Error handling for unreachable or malformed URLs

    The tool returns structured data including page content, metadata, links, and images,
    making it suitable for medical research, fact-checking, and accessing up-to-date
    medical information from authoritative sources.
    """

    name: str = "web_browser"
    description: str = (
        "Searches the web for medical information or visits specific URLs to retrieve content. "
        "Can perform web searches using Google Custom Search API or visit specific medical websites, "
        "journals, and online resources. Returns structured content including text, links, images, "
        "and metadata. Input should be either a search query for web search or a URL for direct access. "
        "Supports configurable content length (default 5000 characters) and link extraction limits (default 5 links). "
        "Useful for accessing current medical research, clinical guidelines, drug information, "
        "and other authoritative online medical resources."
    )
    search_api_key: Optional[str] = None
    search_engine_id: Optional[str] = None
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    max_results: int = 5
    args_schema: Type[BaseModel] = WebBrowserSchema

    def __init__(
        self, search_api_key: Optional[str] = None, search_engine_id: Optional[str] = None, **kwargs
    ):
        """Initialize the web browser tool with optional search API credentials.

        Args:
            search_api_key (Optional[str]): Google Custom Search API key. If not provided,
                                           will attempt to read from GOOGLE_SEARCH_API_KEY environment variable
            search_engine_id (Optional[str]): Google Custom Search Engine ID. If not provided,
                                             will attempt to read from GOOGLE_SEARCH_ENGINE_ID environment variable
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(**kwargs)
        # Try to get API keys from environment variables if not provided
        self.search_api_key = search_api_key or os.environ.get("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = search_engine_id or os.environ.get("GOOGLE_SEARCH_ENGINE_ID")

    def search_web(self, query: str) -> Dict[str, Any]:
        """Search the web using Google Custom Search API.

        Args:
            query (str): The search query string to execute

        Returns:
            Dict[str, Any]: Dictionary containing search results with titles, links, snippets,
                           and source information, or error message if search fails
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
            "num": self.max_results,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()

            if "items" not in results:
                return {"results": [], "message": "No results found"}

            formatted_results = []
            for item in results["items"]:
                formatted_results.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": item.get("displayLink"),
                    }
                )

            return {
                "results": formatted_results,
                "message": f"Found {len(formatted_results)} results for query: {query}",
            }

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    def visit_url(
        self, url: str, max_content_length: int = 5000, max_links: int = 5
    ) -> Dict[str, Any]:
        """Visit a URL and extract its content with comprehensive parsing.

        Args:
            url (str): The URL to visit and parse
            max_content_length (int): Maximum length of text content to extract (default: 5000)
            max_links (int): Maximum number of links to extract (default: 5)

        Returns:
            Dict[str, Any]: Dictionary containing extracted content including:
                - title: Page title
                - content: Cleaned text content (truncated if > max_content_length)
                - url: Original URL
                - links: List of extracted links (limited to max_links)
                - images: List of image URLs (limited to 3)
                - content_type: HTTP content type
                - content_length: Length of extracted text
                - truncated: Boolean indicating if content was truncated
                Or error message if URL access fails
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
            text_content = re.sub(r"\n+", "\n", text_content)
            text_content = re.sub(r" +", " ", text_content)

            # Extract links
            links = []
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # Handle relative URLs
                if href.startswith("/"):
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    href = base_url + href
                if href.startswith(("http://", "https://")):
                    links.append({"text": link.get_text(strip=True) or href, "url": href})

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
                "content": (
                    text_content[:max_content_length]
                    if len(text_content) > max_content_length
                    else text_content
                ),
                "url": url,
                "links": links[:max_links],  # Limit to max_links
                "images": images,
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": len(text_content),
                "truncated": len(text_content) > max_content_length,
            }

        except Exception as e:
            return {"error": f"Failed to visit {url}: {str(e)}"}

    def _run(
        self,
        query: str = "",
        url: str = "",
        max_content_length: int = 5000,
        max_links: int = 5,
        run_manager: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute the web browser tool with the given parameters.

        Args:
            query (str): Search query string (leave empty if visiting a URL)
            url (str): URL to visit (leave empty if performing a search)
            max_content_length (int): Maximum length of text content to extract (default: 5000)
            max_links (int): Maximum number of links to extract (default: 5)
            run_manager (Optional[Any]): Callback manager for the tool run

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
                - output: Dictionary with search results or page content
                - metadata: Dictionary with execution metadata including query, URL, timestamp, and tool name

        Raises:
            Exception: If both query and url are provided or if neither is provided
        """
        metadata = {
            "query": query if query else "",
            "url": url if url else "",
            "max_content_length": max_content_length,
            "max_links": max_links,
            "timestamp": time.time(),
            "tool": "web_browser",
            "operation": "search" if query else "visit_url" if url else "none",
        }

        try:
            if url:
                result = self.visit_url(url, max_content_length, max_links)
                metadata["analysis_status"] = "completed" if "error" not in result else "failed"
                return result, metadata
            elif query:
                result = self.search_web(query)
                metadata["analysis_status"] = "completed" if "error" not in result else "failed"
                return result, metadata
            else:
                error_result = {"error": "Please provide either a search query or a URL to visit"}
                metadata["analysis_status"] = "failed"
                return error_result, metadata

        except Exception as e:
            error_result = {"error": f"Web browser tool failed: {str(e)}"}
            metadata["analysis_status"] = "failed"
            metadata["error_details"] = str(e)
            return error_result, metadata

    async def _arun(
        self,
        query: str = "",
        url: str = "",
        max_content_length: int = 5000,
        max_links: int = 5,
        run_manager: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute the web browser tool asynchronously.

        This method currently calls the synchronous version, as the web requests
        are not inherently asynchronous in this implementation. For true asynchronous
        behavior, consider using aiohttp or similar async HTTP clients.

        Args:
            query (str): Search query string (leave empty if visiting a URL)
            url (str): URL to visit (leave empty if performing a search)
            max_content_length (int): Maximum length of text content to extract (default: 5000)
            max_links (int): Maximum number of links to extract (default: 5)
            run_manager (Optional[Any]): Async callback manager for the tool run

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
                - output: Dictionary with search results or page content
                - metadata: Dictionary with execution metadata

        Raises:
            Exception: If both query and url are provided or if neither is provided
        """
        return self._run(
            query=query,
            url=url,
            max_content_length=max_content_length,
            max_links=max_links,
            run_manager=run_manager,
        )
