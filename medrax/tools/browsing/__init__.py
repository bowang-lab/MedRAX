"""Web browsing tools for MedRAX2 medical agents."""

from .duckduckgo import DuckDuckGoSearchTool, WebSearchInput
from .web_browser import WebBrowserTool, WebBrowserSchema, SearchQuerySchema, VisitUrlSchema

__all__ = [
    "DuckDuckGoSearchTool",
    "WebSearchInput",
    "WebBrowserTool", 
    "WebBrowserSchema",
    "SearchQuerySchema",
    "VisitUrlSchema"
] 