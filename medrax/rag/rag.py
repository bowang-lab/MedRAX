import os
from pathlib import Path
from typing import List
from pydantic import Field

from pydantic import BaseModel, Field
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRerank
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
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
        huggingface_datasets (List[str]): List of HuggingFace dataset names to load (e.g., ["MedRAG/textbooks", "VictorLJZ/medrax"])
        dataset_split (str): Split to use for HuggingFace datasets (default: "train")
    """

    model: str = Field(default="command-a-03-2025")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    pinecone_index_name: str = Field(default="medrax2")
    embedding_model: str = Field(default="embed-v4.0")
    rerank_model: str = Field(default="rerank-v3.5")
    retriever_k: int = Field(default=7)
    chunk_size: int = Field(default=1500)
    chunk_overlap: int = Field(default=300)
    local_docs_dir: str = Field(default="rag_docs")
    huggingface_datasets: List[str] = Field(default_factory=lambda: ["VictorLJZ/medrax2"])
    dataset_split: str = Field(default="train")


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
        self.local_docs_dir = config.local_docs_dir
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="result",
            input_key="query",
        )

        # Initialize Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable not set. Please get a key from app.pinecone.io"
            )
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = self.config.pinecone_index_name

        # Create or update Pinecone index and get the vectorstore
        self.vectorstore = self.get_or_create_vectorstore()

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

    def get_or_create_vectorstore(self) -> PineconeVectorStore:
        """
        Connects to an existing Pinecone index. If the index is empty, it populates it with documents.
        The index must be created manually in the Pinecone console beforehand.
        """
        if self.index_name not in self.pinecone.list_indexes().names():
            raise ValueError(
                f"Index '{self.index_name}' not found in your Pinecone project. "
                f"Please create it manually in the Pinecone console. "
                f"For the free tier, use the 'aws-us-east-1' environment."
            )

        print(f"Connecting to existing Pinecone index: {self.index_name}")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name, embedding=self.embeddings
        )

        # Check if the index is empty and needs to be populated
        try:
            # Get the index object directly
            index = self.pinecone.Index(self.index_name)

            # Get index stats
            stats = index.describe_index_stats()
            print(f"Index stats: {stats}")

            total_vectors = stats.get("total_vector_count", 0)
            print(f"Total vectors in index: {total_vectors}")

            if total_vectors == 0:
                print("Index is empty. Populating with documents...")
                documents = self._load_all_documents()
                if documents:
                    total_docs = len(documents)
                    print(f"Adding {total_docs} documents to the index. This may take a while...")

                    # Batching mechanism to handle rate limits
                    batch_size = 50  # Process 50 documents per batch
                    for i in tqdm(
                        range(0, total_docs, batch_size),
                        desc="Adding documents to Pinecone",
                    ):
                        batch = documents[i : i + batch_size]
                        vectorstore.add_documents(batch)
                        # Removed rate limiting - process as fast as possible

                    print("Documents added successfully.")

                    # Verify documents were added
                    final_stats = index.describe_index_stats()
                    final_count = final_stats.get("total_vector_count", 0)
                    print(f"Final vector count after adding documents: {final_count}")
                else:
                    print("Warning: No documents found to add to the new index.")
            else:
                print(f"Index already populated with {total_vectors} vectors.")

        except Exception as e:
            print(f"Error checking index stats: {e}")
            print("Proceeding without stats check...")

        return vectorstore

    def _load_all_documents(self) -> List[Document]:
        """Collect documents from all enabled sources."""
        all_documents = []

        # Load local documents
        if os.path.exists(self.local_docs_dir):
            print(f"Loading documents from local directory: {self.local_docs_dir}")
            local_docs = self.load_directory(self.local_docs_dir)
            all_documents.extend(local_docs)
            print(f"Loaded {len(local_docs)} documents from local directory")

        # Load HuggingFace datasets
        for dataset_name in self.config.huggingface_datasets:
            print(f"Loading documents from HuggingFace dataset: {dataset_name}...")
            try:
                hf_docs = self.load_huggingface_dataset(dataset_name, self.config.dataset_split)
                all_documents.extend(hf_docs)
                print(f"Loaded {len(hf_docs)} documents from {dataset_name}")
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {str(e)}")
                continue

        if not all_documents:
            print("Warning: No documents loaded. Please check your configuration.")

        return all_documents

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

    def load_huggingface_dataset(self, dataset_name: str, split: str) -> List[Document]:
        """Load MedRAG textbooks dataset from Hugging Face.

        Returns:
            List[Document]: List of processed documents from MedRAG textbooks

        Raises:
            ValueError: If unable to load the dataset
        """
        try:
            print(f"Loading HuggingFace dataset: {dataset_name}...")
            dataset = load_dataset(dataset_name, split=split)
            documents = []

            for item in tqdm(
                dataset,
                desc=f"Processing HuggingFace dataset: {dataset_name}",
                total=len(dataset),
                unit="chunk",
            ):
                # Create a Document object for each textbook snippet
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "source": f"{dataset_name}",
                        "id": item["id"],
                        "title": item["title"],
                    },
                )
                documents.append(doc)

            print(
                f"Loaded {len(documents)} document chunks from HuggingFace dataset: {dataset_name}"
            )
            return documents

        except Exception as e:
            print(f"Error loading HuggingFace dataset: {dataset_name}: {str(e)}")
            raise ValueError(f"Failed to load HuggingFace dataset: {dataset_name}: {str(e)}")

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
