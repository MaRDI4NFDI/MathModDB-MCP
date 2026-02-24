from abc import ABC, abstractmethod
from typing import Literal, Union, overload

from fastembed import LateInteractionTextEmbedding
from openai import OpenAI
from qdrant_client import models
from sentence_transformers import SentenceTransformer

MultivectorOutput = list[list[float]]
BatchMultivectorOutput = list[MultivectorOutput]

DenseOutput = list[float]
BatchDenseOutput = list[DenseOutput]


class TextEmbedder(ABC):
    """
    Abstract base class for text embedding providers.

    This class defines the interface that all text embedding providers must implement.
    It supports both single text and batch text embedding operations with proper type hints.
    """

    @overload
    def embed(self, texts: str) -> DenseOutput:
        """Generate embedding for a single text string."""
        ...

    @overload
    def embed(self, texts: list[str]) -> BatchDenseOutput:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def embed(
        self,
        texts: Union[list[str], str],
    ) -> Union[DenseOutput, BatchDenseOutput]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            If a single string is provided: a single embedding vector (list of floats)
            If a list is provided: list of embedding vectors (list of lists of floats)
        """
        raise NotImplementedError("Subclasses must implement embed method")

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """
        Expected dimension of the embedding vectors.

        Returns:
            Integer representing the dimension size
        """
        raise NotImplementedError("Subclasses must implement embed_dim method")

    @property
    @abstractmethod
    def vector_params(self) -> models.VectorParams:
        """
        Vector parameters for the embedder.

        Returns:
            VectorParams object containing configuration for vector storage,
            including size, distance metric, and other vector-specific settings.
        """
        raise NotImplementedError("Subclasses must implement embed method")

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the embedder.

        Returns:
            String identifier for the embedding model or provider.
        """
        raise NotImplementedError("Subclasses must implement name method")


class MultivectorEmbedder(TextEmbedder):
    """
    Multivector embedding provider.

    This class extends TextEmbedder to support multivector embeddings,
    where each text input produces multiple embedding vectors instead of a single one.
    This is useful for models like ColBERT that generate token-level embeddings.
    """

    @overload
    def embed(self, texts: str) -> MultivectorOutput:
        """Generate embedding for a single text string."""
        ...

    @overload
    def embed(self, texts: list[str]) -> BatchMultivectorOutput:
        """Generate embeddings for a list of texts."""
        ...

    def embed(
        self,
        texts: Union[list[str], str],
    ) -> Union[MultivectorOutput, BatchMultivectorOutput]:
        """
        Generate multivector embeddings for a list of texts.

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            If a single string is provided: list of embedding vectors for that text
            If a list is provided: list of lists of embedding vectors, one per input text
        """
        raise NotImplementedError("Subclasses must implement embed method")


class OpenAIEmbedder(TextEmbedder):
    """
    OpenAI embedding provider using text-embedding-3 models.

    This class provides access to OpenAI's text-embedding-3-large and text-embedding-3-small
    models through the OpenAI API. It supports configurable embedding dimensions and
    handles both single text and batch embedding operations.
    """

    def __init__(
        self,
        model: Literal[
            "text-embedding-3-large", "text-embedding-3-small"
        ] = "text-embedding-3-large",
        dimensions: int = 3072,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name. Choose from:
                - "text-embedding-3-large": Higher quality, 3072 dimensions by default
                - "text-embedding-3-small": Faster and cheaper, 1536 dimensions by default
            dimensions: Desired embedding dimension. Must be compatible with the chosen model.
                For text-embedding-3-large: up to 3072 dimensions
                For text-embedding-3-small: up to 1536 dimensions
        """
        self.model = model
        self.dimensions = dimensions
        self.client = OpenAI()

    @overload
    def embed(self, texts: str) -> list[float]:
        """Generate embedding for a single text string using OpenAI API."""
        ...

    @overload
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using OpenAI API."""
        ...

    def embed(
        self, texts: Union[list[str], str]
    ) -> Union[list[float], list[list[float]]]:
        """
        Generate embeddings using OpenAI API.

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            If a single string is provided: a single embedding vector
            If a list is provided: list of embedding vectors, one per input text

        Raises:
            OpenAI API exceptions for authentication, rate limiting, or other API errors
        """
        if isinstance(texts, str):
            response = self.client.embeddings.create(
                input=[texts],
                model=self.model,
                dimensions=self.dimensions,
            )
            return response.data[0].embedding

        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self.dimensions,
        )
        return [data.embedding for data in response.data]

    @property
    def embed_dim(self) -> int:
        """
        Return the configured embedding dimension.

        Returns:
            Integer representing the dimension size configured for this embedder
        """
        return self.dimensions

    @property
    def name(self) -> str:
        """
        Return the name of the embedder.

        Returns:
            String identifier of the OpenAI model being used
        """
        return self.model

    @property
    def vector_params(self) -> models.VectorParams:
        """
        Return the vector parameters for the embedder.

        Returns:
            VectorParams configured for OpenAI embeddings with cosine distance metric
        """
        return models.VectorParams(
            size=self.embed_dim,
            distance=models.Distance.COSINE,
        )


class HuggingFaceEmbedder(TextEmbedder):
    """
    HuggingFace embedding provider using SentenceTransformers.

    This class provides access to any SentenceTransformer-compatible model from
    HuggingFace Hub. It automatically detects the embedding dimension and handles
    both single text and batch embedding operations.
    """

    def __init__(
        self, model_name: str = "math-similarity/Bert-MLM_arXiv-MP-class_zbMath"
    ):
        """
        Initialize HuggingFace embedder.

        Args:
            model_name: HuggingFace model name or path. Can be:
                - A model ID from HuggingFace Hub (e.g., "sentence-transformers/all-MiniLM-L6-v2")
                - A local path to a saved model
                - Default: "math-similarity/Bert-MLM_arXiv-MP-class_zbMath" (specialized for math content)

        Raises:
            ValueError: If the model doesn't have a defined embedding dimension
        """
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(
                f"Model {model_name} does not have a defined embedding dimension"
            )
        self._dimension = dimension

    @overload
    def embed(self, texts: str) -> list[float]:
        """Generate embedding for a single text string using SentenceTransformer."""
        ...

    @overload
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using SentenceTransformer."""
        ...

    def embed(
        self, texts: Union[list[str], str]
    ) -> Union[list[float], list[list[float]]]:
        """
        Generate embeddings using SentenceTransformer.

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            If a single string is provided: a single embedding vector
            If a list is provided: list of embedding vectors, one per input text

        Note:
            All embeddings are converted to Python floats for consistency
        """
        if isinstance(texts, str):
            emb = self._model.encode(texts, convert_to_numpy=False)
            return [float(x) for x in emb]

        embeddings = self._model.encode(texts, convert_to_numpy=False)
        return [[float(x) for x in emb] for emb in embeddings]

    @property
    def embed_dim(self) -> int:
        """
        Return the model's embedding dimension.

        Returns:
            Integer representing the dimension size of the loaded model
        """
        return self._dimension

    @property
    def name(self) -> str:
        """
        Return the name of the embedder.

        Returns:
            String identifier of the HuggingFace model being used
        """
        return self.model_name

    @property
    def vector_params(self) -> models.VectorParams:
        """
        Return the vector parameters for the embedder.

        Returns:
            VectorParams configured for HuggingFace embeddings with cosine distance metric
        """
        return models.VectorParams(
            size=self.embed_dim,
            distance=models.Distance.COSINE,
        )


class ColBERTEmbedder(MultivectorEmbedder):
    """
    ColBERT embedding provider for late interaction retrieval.

    ColBERT (Contextualized Late Interaction over BERT) generates multiple embedding
    vectors per text (one per token), enabling more fine-grained similarity matching.
    This is particularly effective for information retrieval tasks.
    """

    def __init__(
        self,
        model_name: Literal[
            "colbert-ir/colbertv2.0",
            "answerdotai/answerai-colbert-small-v1",
        ] = "colbert-ir/colbertv2.0",
    ):
        """
        Initialize ColBERT embedder.

        Args:
            model_name: ColBERT model name. Choose from:
                - "colbert-ir/colbertv2.0": Original ColBERT v2 model, high quality
                - "answerdotai/answerai-colbert-small-v1": Smaller, faster variant

        Raises:
            ValueError: If the model doesn't have a defined embedding dimension
        """
        self.model_name = model_name
        self._model = LateInteractionTextEmbedding(model_name)
        dimension = self._model.get_embedding_size(model_name)
        if dimension is None:
            raise ValueError(
                f"Model {model_name} does not have a defined embedding dimension"
            )
        self._dimension = dimension

    @overload
    def embed(self, texts: str) -> MultivectorOutput:
        """Generate embedding for a single text string using ColBERT."""
        ...

    @overload
    def embed(self, texts: list[str]) -> BatchMultivectorOutput:
        """Generate embeddings for a list of texts using ColBERT."""
        ...

    def embed(
        self,
        texts: Union[list[str], str],
    ) -> Union[MultivectorOutput, BatchMultivectorOutput]:
        """
        Generate multivector embeddings using ColBERT.

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            If a single string is provided: list of embedding vectors (one per token)
            If a list is provided: list of lists of embedding vectors, one per input text

        Note:
            Each text produces multiple embedding vectors corresponding to its tokens.
            The number of vectors varies based on text length and tokenization.
        """
        if isinstance(texts, str):
            emb = self._model.embed(texts)
            emb = [x.tolist() for x in emb]
            return emb[0]

        embeddings = self._model.embed(texts)

        return [x.tolist() for x in embeddings]

    @property
    def embed_dim(self) -> int:
        """
        Return the model's embedding dimension.

        Returns:
            Integer representing the dimension size of each individual embedding vector
        """
        return self._dimension

    @property
    def name(self) -> str:
        """
        Return the name of the embedder.

        Returns:
            String identifier of the ColBERT model being used
        """
        return self.model_name

    @property
    def vector_params(self) -> models.VectorParams:
        """
        Return the vector parameters for the embedder.

        Returns:
            VectorParams configured for ColBERT multivector embeddings with:
            - Cosine distance metric
            - MAX_SIM comparator for late interaction
            - Optimized HNSW configuration for multivector search
        """
        return models.VectorParams(
            size=self.embed_dim,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
        )
