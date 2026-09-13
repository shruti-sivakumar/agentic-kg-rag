"""
Shared sentence-embedding encoder.

A single sentence-transformer model is used for every embedding
computation in the system: dense passage retrieval, entity-linking
disambiguation, and knowledge-graph edge scoring. Using one encoder
instance for all three keeps the resulting similarity scores on a common
scale and avoids loading model weights more than once.

The default model, `all-MiniLM-L6-v2`, is chosen for its small memory
footprint (384-dimensional output, on the order of tens of megabytes of
weights), which suits the target deployment environment of a single
consumer-grade GPU.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=None)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_encoder(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Return the shared encoder instance for `model_name`.

    Repeated calls with the same model name return the same object, so
    that all consumers draw on one loaded copy of the model.
    """
    return _load_model(model_name)


def encode(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode a batch of strings into L2-normalized embeddings.

    Embeddings are normalized to unit length so that inner product is
    equivalent to cosine similarity, matching how they are consumed by
    the FAISS index (inner-product search) and by edge scoring in the
    knowledge-graph traversal step (cosine similarity between a candidate
    triple's verbalization and the question).
    """
    encoder = get_encoder(model_name)
    embeddings = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def embedding_dimension(model_name: str = DEFAULT_MODEL_NAME) -> int:
    return get_encoder(model_name).get_embedding_dimension()
