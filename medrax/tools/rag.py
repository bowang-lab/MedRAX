from langchain.tools import BaseTool
from medrax.rag.rag import RAGConfig, CohereRAG
from langchain.chains import RetrievalQA
from typing import Dict, Tuple, Any


class RAGTool(BaseTool):
    """Tool for answering medical questions using RAG with comprehensive medical knowledge base.

    This tool leverages Retrieval-Augmented Generation (RAG) to answer medical questions by
    retrieving relevant information from a curated knowledge base of medical textbooks, research papers,
    and clinical manuals. The tool uses advanced embedding models to find contextually relevant
    information and then generates comprehensive, evidence-based answers using large language models.

    The knowledge base includes:
    - Medical textbooks and reference materials
    - Research papers and clinical studies
    - Medical manuals and guidelines
    - Specialized medical literature

    Args:
        config (RAGConfig): Configuration object containing model settings, embedding model,
                          knowledge base paths, and other RAG system parameters
    """

    name: str = "medical_knowledge_rag"
    description: str = (
        "Answers medical questions using Retrieval-Augmented Generation with a comprehensive medical knowledge base. "
        "Retrieves relevant information from medical textbooks, research papers, and clinical manuals, "
        "then generates evidence-based answers using advanced language models. "
        "Input should be a medical question or query in natural language. "
        "Output includes a comprehensive answer with supporting source documents and metadata. "
    )
    rag: CohereRAG = None
    chain: RetrievalQA = None

    def __init__(
        self,
        config: RAGConfig,
    ):
        """Initialize RAG tool with configuration.

        Args:
            config (RAGConfig): Configuration for the RAG system including model settings,
                               embedding model, knowledge base paths, and retrieval parameters
        """
        super().__init__()
        self.rag = CohereRAG(config)
        self.chain = self.rag.initialize_rag(with_memory=True)

    def _run(self, query: str) -> Tuple[Dict[str, Any], Dict]:
        """Execute the RAG tool with the given query.

        Args:
            query (str): Medical question to answer

        Returns:
            Tuple[Dict[str, Any], Dict]: Output dictionary and metadata dictionary
        """
        try:
            result = self.chain.invoke({"query": query})

            output = {
                "answer": result["result"],
                "source_documents": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in result.get("source_documents", [])
                ],
            }

            metadata = {
                "query": query,
                "analysis_status": "completed",
                "num_sources": len(result.get("source_documents", [])),
                "model": self.rag.config.model,
                "embedding_model": self.rag.config.embedding_model,
            }

            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "query": query,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(self, query: str) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run.

        Args:
            query (str): Medical question to answer

        Returns:
            Tuple[Dict[str, Any], Dict]: Output dictionary and metadata dictionary

        Raises:
            NotImplementedError: Async not implemented yet
        """
        return self._run(query)
