"""
Citation management system for MedRAX2.

This module handles citation extraction from agent responses and matching
with source information from tool outputs.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Citation:
    """Represents a citation with source information."""
    
    number: int
    title: str
    url: Optional[str] = None
    source_type: str = "unknown"  # "rag", "web", "search"
    snippet: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CitationManager:
    """Manages citation extraction and source matching."""
    
    def __init__(self):
        self.citations: List[Citation] = []
        self.citation_counter = 0
    
    def extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from text like [1], [2], etc.
        
        Args:
            text (str): Text to extract citations from
            
        Returns:
            List[int]: List of citation numbers found
        """
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return [int(match) for match in matches]
    
    def create_citations_from_tool_outputs(self, tool_messages: List[Dict[str, Any]]) -> List[Citation]:
        """Create citations from recent tool outputs.
        
        Args:
            tool_messages (List[Dict[str, Any]]): List of tool messages from conversation
            
        Returns:
            List[Citation]: List of created citations
        """
        citations = []
        citation_num = 1
        
        for msg in tool_messages:
            if not isinstance(msg, dict):
                continue
                
            # Handle RAG tool output
            if "source_documents" in msg:
                for doc in msg.get("source_documents", []):
                    if isinstance(doc, dict):
                        title = doc.get("metadata", {}).get("title", "Unknown Source")
                        url = doc.get("metadata", {}).get("source", "")
                        snippet = doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                        
                        citation = Citation(
                            number=citation_num,
                            title=title,
                            url=url,
                            source_type="rag",
                            snippet=snippet,
                            metadata=doc.get("metadata", {})
                        )
                        citations.append(citation)
                        citation_num += 1
            
            # Handle web browser tool output
            elif "results" in msg:
                for result in msg.get("results", []):
                    if isinstance(result, dict):
                        citation = Citation(
                            number=citation_num,
                            title=result.get("title", "Web Search Result"),
                            url=result.get("link", ""),
                            source_type="web",
                            snippet=result.get("snippet", ""),
                            metadata={"source": result.get("source", "")}
                        )
                        citations.append(citation)
                        citation_num += 1
            
            # Handle direct URL visit
            elif "title" in msg and "url" in msg:
                citation = Citation(
                    number=citation_num,
                    title=msg.get("title", "Web Page"),
                    url=msg.get("url", ""),
                    source_type="web",
                    snippet=msg.get("content", "")[:200] + "..." if len(msg.get("content", "")) > 200 else msg.get("content", ""),
                    metadata={"content_type": msg.get("content_type", "")}
                )
                citations.append(citation)
                citation_num += 1
        
        return citations
    
    def format_citations_for_display(self, citations: List[Citation]) -> List[Dict[str, Any]]:
        """Format citations for frontend display.
        
        Args:
            citations (List[Citation]): List of citations to format
            
        Returns:
            List[Dict[str, Any]]: Formatted citation data for display
        """
        formatted = []
        
        for citation in citations:
            formatted.append({
                "number": citation.number,
                "title": citation.title,
                "url": citation.url,
                "source_type": citation.source_type,
                "snippet": citation.snippet,
                "metadata": citation.metadata
            })
        
        return formatted
    
    def match_citations_to_text(self, text: str, available_citations: List[Citation]) -> Tuple[str, List[Citation]]:
        """Match citation numbers in text to available citations.
        
        Args:
            text (str): Text containing citation markers
            available_citations (List[Citation]): Available citations to match
            
        Returns:
            Tuple[str, List[Citation]]: (original text, matched citations)
        """
        cited_numbers = self.extract_citations(text)
        matched_citations = []
        
        for num in cited_numbers:
            # Find citation with matching number
            for citation in available_citations:
                if citation.number == num:
                    matched_citations.append(citation)
                    break
        
        return text, matched_citations 