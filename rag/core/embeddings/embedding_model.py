from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.embeddings.llamacpp import LlamaCppEmbeddings


def load_embedding(name: str) -> Embeddings:
    if name == "openai":
        return OpenAIEmbeddings()
    elif name == "llama":
        return LlamaCppEmbeddings()
    else:
        raise Exception("Unknown Embeddings model: " + name)
    