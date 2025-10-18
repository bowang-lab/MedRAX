"""
Web search tool for MedRAX2 medical agents.

Provides DuckDuckGo search capabilities for medical agents to retrieve
real-time information from the web with proper error handling
and result formatting. Designed specifically for medical research,
fact-checking, and accessing current medical information.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Tuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(
        ...,
        description="The search query to look up on the web. Be specific and include relevant medical keywords for better results.",
        min_length=1,
        max_length=500,
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of search results to return (1-10)",
        ge=1,
        le=10,
    )
    region: str = Field(
        default="us-en",
        description="Region for search results (e.g., 'us-en', 'uk-en', 'ca-en')",
    )


class DuckDuckGoSearchTool(BaseTool):
    """
    Tool that performs web searches using DuckDuckGo search engine for medical research.

    This tool provides access to real-time web information through DuckDuckGo's
    search API, specifically designed for medical agents that need to retrieve current
    medical information, verify facts, or find resources on medical topics.

    Features:
        - Real-time web search capability for medical information
        - Configurable number of results (1-10)
        - Regional search support for localized medical results
        - Robust error handling for network issues
        - Structured result formatting for easy parsing
        - Privacy-focused (DuckDuckGo doesn't track users)
        - Medical-focused search optimization

    Use Cases:
        - Medical fact checking and verification
        - Finding current medical news and updates
        - Researching specific medical topics or questions
        - Gathering multiple perspectives on medical issues
        - Locating official medical resources and documentation
        - Accessing current clinical guidelines and research

    Rate Limiting:
        DuckDuckGo has rate limits. Avoid making too many rapid requests
        to prevent temporary blocking.
    """

    name: str = "duckduckgo_search"
    description: str = (
        "Search the web using DuckDuckGo to find current medical information, research, and resources. "
        "Input should be a clear search query with relevant medical keywords. The tool returns a list of relevant web results "
        "with titles, URLs, and brief snippets. Useful for medical fact-checking, finding current medical events, "
        "researching medical topics, and gathering information from reliable medical sources. "
        "Results are privacy-focused and don't track user searches. Optimized for medical research and clinical information."
    )
    args_schema: type[BaseModel] = WebSearchInput
    return_direct: bool = False

    def __init__(self, **kwargs):
        """Initialize the DuckDuckGo search tool."""
        super().__init__(**kwargs)

        if DDGS is None:
            logger.error("duckduckgo-search package not installed. Install with: pip install duckduckgo-search")
            raise ImportError("duckduckgo-search package is required for web search functionality")

        logger.info("DuckDuckGo search tool initialized successfully")

    def _perform_search_sync(self, query: str, max_results: int = 5, region: str = "us-en") -> Dict[str, Any]:
        """
        Perform the actual web search using DuckDuckGo synchronously.

        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return.
            region (str): Region for localized results.

        Returns:
            Dict[str, Any]: Structured search results.
        """
        logger.info(f"Performing web search: '{query}' (max_results={max_results}, region={region})")

        try:
            # Initialize DDGS with error handling
            with DDGS() as ddgs:
                # Perform the search
                search_results = list(
                    ddgs.text(
                        keywords=query,
                        region=region,
                        safesearch="moderate",
                        timelimit=None,
                        max_results=max_results,
                    )
                )

                # Format results for the agent
                formatted_results = []
                for i, result in enumerate(search_results, 1):
                    formatted_result = {
                        "rank": i,
                        "title": result.get("title", "No title"),
                        "url": result.get("href", "No URL"),
                        "snippet": result.get("body", "No description available"),
                        "source": "DuckDuckGo",
                    }
                    formatted_results.append(formatted_result)

                # Create summary for the agent
                if formatted_results:
                    summary = (
                        f"Found {len(formatted_results)} results for '{query}'. Top results include: "
                        + ", ".join([f"{r['title']}" for r in formatted_results[:3]])
                    )
                else:
                    summary = f"No results found for '{query}'"

                # Log successful completion
                logger.info(f"Web search completed successfully: {len(formatted_results)} results")

                return {
                    "query": query,
                    "results_count": len(formatted_results),
                    "results": formatted_results,
                    "summary": summary,
                    "search_engine": "DuckDuckGo",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            error_msg = f"Web search failed for query '{query}': {str(e)}"
            logger.error(f"{error_msg}")

            return {
                "query": query,
                "results_count": 0,
                "results": [],
                "error": error_msg,
                "search_engine": "DuckDuckGo",
                "timestamp": datetime.now().isoformat(),
            }

    def _run(
        self,
        query: str,
        max_results: int = 5,
        region: str = "us-en",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute the web search synchronously.

        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            region (str): Search region
            run_manager: Callback manager (unused)

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
                - output: Dictionary with search results
                - metadata: Dictionary with execution metadata
        """
        # Create metadata structure
        metadata = {
            "query": query,
            "max_results": max_results,
            "region": region,
            "timestamp": time.time(),
            "tool": "duckduckgo_search",
            "operation": "search",
        }

        try:
            result = self._perform_search_sync(query, max_results, region)

            # Check if search was successful
            if "error" in result:
                metadata["analysis_status"] = "failed"
                metadata["error_details"] = result["error"]
            else:
                metadata["analysis_status"] = "completed"
                metadata["results_count"] = result.get("results_count", 0)

            return result, metadata

        except Exception as e:
            error_result = {
                "query": query,
                "results_count": 0,
                "results": [],
                "error": str(e),
                "search_engine": "DuckDuckGo",
                "timestamp": datetime.now().isoformat(),
            }
            metadata["analysis_status"] = "failed"
            metadata["error_details"] = str(e)

            return error_result, metadata

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        region: str = "us-en",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute the web search asynchronously.

        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            region (str): Search region
            run_manager: Callback manager (unused)

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
                - output: Dictionary with search results
                - metadata: Dictionary with execution metadata
        """
        # Try to get LangGraph stream writer for progress updates
        writer = None
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            # Stream writer not available (outside LangGraph context)
            pass

        if writer:
            writer(
                {
                    "tool_name": "DuckDuckGoSearchTool",
                    "status": "started",
                    "query": query,
                    "max_results": max_results,
                    "step": "Initiating web search",
                }
            )

        try:
            if writer:
                writer(
                    {
                        "tool_name": "DuckDuckGoSearchTool",
                        "status": "searching",
                        "step": "Fetching results from DuckDuckGo API",
                    }
                )

            # Use asyncio to run sync search in executor
            loop = asyncio.get_event_loop()
            result, metadata = await loop.run_in_executor(None, self._run, query, max_results, region)

            if writer:
                # Parse result to get count for progress update
                results_count = result.get("results_count", 0)
                writer(
                    {
                        "tool_name": "DuckDuckGoSearchTool",
                        "status": "completed",
                        "step": f"Search completed with {results_count} results",
                        "results_count": results_count,
                    }
                )

            return result, metadata

        except Exception as e:
            if writer:
                writer(
                    {
                        "tool_name": "DuckDuckGoSearchTool",
                        "status": "error",
                        "step": f"Search failed: {str(e)}",
                        "error": str(e),
                    }
                )

            error_result = {
                "query": query,
                "results_count": 0,
                "results": [],
                "error": str(e),
                "search_engine": "DuckDuckGo",
                "timestamp": datetime.now().isoformat(),
            }

            metadata = {
                "query": query,
                "max_results": max_results,
                "region": region,
                "timestamp": time.time(),
                "tool": "duckduckgo_search",
                "operation": "search",
                "analysis_status": "failed",
                "error_details": str(e),
            }

            return error_result, metadata

    def get_search_summary(self, query: str, max_results: int = 3) -> dict[str, str | list[str]]:
        """
        Get a quick summary of search results for a given query.

        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to summarize.

        Returns:
            Dict[str, Union[str, List[str]]]: Summary of search results.
        """
        try:
            result, _ = self._run(query, max_results)

            if "error" in result:
                return {
                    "query": query,
                    "status": "error",
                    "error": result["error"],
                    "results": [],
                }

            # Extract key information
            results = result.get("results", [])
            titles = [r["title"] for r in results]
            urls = [r["url"] for r in results]
            snippets = [(r["snippet"][:100] + "..." if len(r["snippet"]) > 100 else r["snippet"]) for r in results]

            return {
                "query": query,
                "status": "success",
                "total_results": result.get("results_count", 0),
                "titles": titles,
                "urls": urls,
                "snippets": snippets,
            }

        except Exception as e:
            logger.error(f"Error getting search summary: {e}")
            return {
                "query": query,
                "status": "error",
                "error": str(e),
                "results": [],
            }
