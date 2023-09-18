from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.base import Embeddings
import pinecone
from decouple import config

PINECONE_API_KEY = config("PINECONE_API_KEY", cast=str)
PINECONE_ENVIRONMENT = config(
    "PINECONE_ENVIRONMENT",
    default="gcp-starter", cast=str
)


def load_vector_store(
        vector_store: str,
        embed: Embeddings, 
        conf: dict = None,
)  -> VectorStore:
    if vector_store == "pinecone":
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        index = pinecone.Index("rag-demo")
        return Pinecone(index, embed.embed_query, "text")
    else:
        raise Exception("Unknown vector store: " + vector_store)