# Citation Feature Implementation

## Overview
The MedRAX2 system now includes automatic citation support, similar to Google's Gemini. When the agent provides information based on tool outputs (RAG or web search), it automatically generates numbered citations that link to the original sources.

## How It Works

### 1. Agent Citations
The system prompt has been updated to instruct the agent to:
- Include numbered citations `[1]`, `[2]`, `[3]` when referencing tool outputs
- Cite sources immediately after making claims
- Maintain consistent citation numbering

### 2. Citation Extraction
- The system automatically extracts citation numbers from agent responses
- Matches citations to recent tool outputs in the conversation
- Supports both RAG and web browser tool outputs

### 3. UI Display
- Citations appear in a dedicated "Sources & Citations" panel
- Each citation shows:
  - Source title
  - Clickable URL (opens in new tab)
  - Snippet preview
  - Source type indicator (📄 for RAG, 🔗 for web)

## Implementation Details

### Files Modified
- `medrax/docs/system_prompts.txt` - Added citation instructions
- `medrax/utils/citations.py` - New citation management system
- `medrax/utils/__init__.py` - Added citation imports
- `interface.py` - Updated chat interface with citation display

### Key Components

#### CitationManager Class
- Extracts citation numbers from text using regex
- Creates citations from tool outputs
- Matches citations to agent responses
- Formats citations for display

#### Citation Data Structure
```python
@dataclass
class Citation:
    number: int
    title: str
    url: Optional[str] = None
    source_type: str = "unknown"  # "rag", "web", "search"
    snippet: Optional[str] = None
    metadata: Dict[str, Any] = None
```

### Supported Tool Outputs

#### RAG Tool
- Extracts source documents with titles and URLs
- Shows content snippets
- Preserves metadata from original sources

#### Web Browser Tool
- Supports search results with titles and links
- Handles direct URL visits
- Shows page content snippets

## Usage Example

**Agent Response:**
```
According to recent research [1], chest X-rays can effectively diagnose pneumonia. 
The Mayo Clinic guidelines [2] recommend specific imaging protocols for optimal results.
```

**Citation Display:**
```
📚 Sources & Citations

📄 [1] Chest X-ray Diagnostics
https://medical-journal.com/chest-xray
Chest X-rays are important for diagnosing pneumonia...

🔗 [2] Mayo Clinic - Pneumonia
https://mayoclinic.org/pneumonia
Pneumonia is an infection that inflames air sacs...
```

## Benefits

1. **Credibility**: Users can verify information sources
2. **Transparency**: Clear indication of information sources
3. **Medical Standards**: Meets medical documentation requirements
4. **User Experience**: Easy access to original sources
5. **Automatic**: No manual citation management required

## Future Enhancements

- Support for more source types (papers, PDFs, etc.)
- Citation export functionality
- Advanced citation formatting options
- Integration with medical databases 