import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import Field

from pydantic import BaseModel, Field
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRerank
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from typing import Callable
from datasets import load_dataset
from tqdm import tqdm


class RAGConfig(BaseModel):
    """Configuration class for RAG (Retrieval Augmented Generation) system.

    Attributes:
        model (str): Cohere model name for chat completion
        temperature (float): Sampling temperature between 0 and 1
        persist_dir (str): Directory to persist vector database
        embedding_model (str): Cohere model name for embeddings
        rerank_model (str): Cohere model name for reranking
        retriever_k (int): Number of documents to retrieve
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Overlap between text chunks
        local_docs_dir (str): Directory for text files
        use_medrag_textbooks (bool): Whether to use MedRAG textbooks dataset
    """

    model: str = Field(default="command-a-03-2025")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    persist_dir: str = Field(default="vector_database")
    embedding_model: str = Field(default="embed-english-v3.0")
    rerank_model: str = Field(default="rerank-v3.5")
    retriever_k: int = Field(default=2)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    local_docs_dir: str = Field(default="medrax/rag/docs")
    use_medrag_textbooks: bool = Field(default=True)


class RerankingRetriever(BaseRetriever):
    """Custom retriever that wraps a document retrieval function with reranking.

    Attributes:
        get_relevant_docs_func (Callable): Function that retrieves relevant documents
    """

    get_relevant_docs_func: Callable[[str], List[Document]] = Field(...)

    def __init__(self, get_relevant_docs_func: Callable[[str], List[Document]]):
        """Initialize retriever with document retrieval function."""
        super().__init__(get_relevant_docs_func=get_relevant_docs_func)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query (str): Search query

        Returns:
            List[Document]: Retrieved documents
        """
        return self.get_relevant_docs_func(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async retrieval not implemented."""
        raise NotImplementedError("Async retrieval not implemented")


class CohereRAG:
    """RAG system implementation using Cohere's models for embedding, reranking and chat.

    Attributes:
        config (RAGConfig): Configuration for the RAG system
        chat_model (ChatCohere): Cohere chat model
        embeddings (CohereEmbeddings): Cohere embeddings model
        reranker (CohereRerank): Cohere reranking model
        persist_dir (str): Directory for vector database
        memory (ConversationBufferMemory): Conversation memory
        vectorstore (Optional[Chroma]): Vector database for document storage
        local_docs_dir (str): Directory for text files
    """

    def __init__(self, config: RAGConfig = RAGConfig()):
        """Initialize RAG system with given configuration."""
        self.config = config
        self.chat_model = ChatCohere(model=config.model, temperature=config.temperature)
        self.embeddings = CohereEmbeddings(model=config.embedding_model)
        self.reranker = CohereRerank(model=config.rerank_model)
        self.persist_dir = config.persist_dir
        self.local_docs_dir = config.local_docs_dir
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="result",
            input_key="query",
        )
        self.vectorstore = self.load_or_create_vectorstore()

        # Initialize vectorstore if empty
        if self.vectorstore is None:
            # Collect documents from all enabled sources
            all_documents = []

            # Load MedRAG textbooks if enabled
            if self.config.use_medrag_textbooks:
                print("Loading documents from MedRAG textbooks...")
                medrag_docs = self.load_medrag_textbooks()
                all_documents.extend(medrag_docs)
                print(f"Loaded {len(medrag_docs)} documents from MedRAG textbooks")

            # Load local documents if enabled
            if os.path.exists(self.local_docs_dir):
                print(f"Loading documents from local directory: {self.local_docs_dir}")
                local_docs = self.load_directory(self.local_docs_dir)
                all_documents.extend(local_docs)
                print(f"Loaded {len(local_docs)} documents from local directory")

            # Create vectorstore with all documents
            if all_documents:
                print(f"Creating vectorstore with {len(all_documents)} total documents")
                self.create_or_update_vectorstore(all_documents)
            else:
                print("Warning: No documents loaded. Please check your configuration.")

    def load_directory(self, directory_path: str) -> List[Document]:
        """Load and split all .txt files from a directory into documents.

        Args:
            directory_path (str): Path to directory containing text files

        Returns:
            List[Document]: List of processed documents

        Raises:
            ValueError: If directory does not exist
        """
        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")

        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )

        # Process each document file (txt, pdf, docx)
        for file_pattern in ["**/*.txt", "**/*.pdf", "**/*.docx"]:
            for file_path in directory.glob(file_pattern):
                try:
                    # Select appropriate loader based on file extension
                    if file_path.suffix.lower() == ".txt":
                        loader = TextLoader(str(file_path))
                    elif file_path.suffix.lower() == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == ".docx":
                        loader = Docx2txtLoader(str(file_path))

                    docs = loader.load()

                    # Split documents first
                    split_docs = text_splitter.split_documents(docs)

                    # Add metadata to each document
                    metadata = {
                        "source": str(file_path),
                        "created_at": os.path.getctime(file_path),
                        "file_type": file_path.suffix.lower()[1:],
                    }

                    for doc in split_docs:
                        doc.metadata.update(metadata)

                    documents.extend(split_docs)
                    print(f"Loaded and split: {file_path}")

                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

        return documents

    def load_or_create_vectorstore(self) -> Optional[Chroma]:
        """Load existing vectorstore or prepare for new one.

        Returns:
            Optional[Chroma]: Loaded vectorstore or None if not exists
        """
        if os.path.exists(self.persist_dir):
            print("Loading existing vectorstore...")
            return Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        return None

    def create_or_update_vectorstore(self, documents: List[Document]):
        """Create new vectorstore or add documents to existing one.

        Args:
            documents (List[Document]): Documents to add to vectorstore
        """
        if self.vectorstore is None:
            print("Creating new vectorstore...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
            )
        else:
            print("Adding documents to existing vectorstore...")
            self.vectorstore.add_documents(documents)

        print(f"Vectorstore saved to {self.persist_dir}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using vector similarity and reranking.

        Args:
            query (str): Search query

        Returns:
            List[Document]: Reranked relevant documents
        """
        # Get initial candidates using vector similarity
        docs = self.vectorstore.similarity_search(query, k=self.config.retriever_k * 2)

        # Rerank documents
        reranked = self.reranker.rerank(query=query, documents=[doc.page_content for doc in docs])

        # Return top k documents after reranking
        return [docs[result["index"]] for result in reranked[: self.config.retriever_k]]

    def initialize_rag(self, with_memory: bool = False) -> RetrievalQA:
        """Initialize RAG chain with optional conversation memory.

        Args:
            with_memory (bool): Whether to include conversation memory

        Returns:
            RetrievalQA: Configured RAG chain

        Raises:
            ValueError: If vectorstore not initialized
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please add documents first.")

        # Create custom retriever
        retriever = RerankingRetriever(self.get_relevant_documents)

        # Configure chain parameters
        chain_kwargs = {
            "llm": self.chat_model,
            "chain_type": "stuff",
            "retriever": retriever,
            "return_source_documents": True,
            "verbose": True,
        }

        if with_memory:
            chain_kwargs["memory"] = self.memory

        return RetrievalQA.from_chain_type(**chain_kwargs)

    def load_medrag_textbooks(self) -> List[Document]:
        """Load MedRAG textbooks dataset from Hugging Face.

        Returns:
            List[Document]: List of processed documents from MedRAG textbooks

        Raises:
            ValueError: If unable to load the dataset
        """
        try:
            print("Loading MedRAG textbooks dataset...")
            dataset = load_dataset("MedRAG/textbooks", split="train")
            documents = []

            for item in tqdm(
                dataset, desc="Processing MedRAG textbooks", total=len(dataset), unit="chunk"
            ):
                # Create a Document object for each textbook snippet
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "source": f"MedRAG/textbooks",
                        "id": item["id"],
                        "title": item["title"],
                    },
                )
                documents.append(doc)

            print(f"Loaded {len(documents)} document chunks from MedRAG textbooks")
            return documents

        except Exception as e:
            print(f"Error loading MedRAG textbooks: {str(e)}")
            raise ValueError(f"Failed to load MedRAG textbooks dataset: {str(e)}")
