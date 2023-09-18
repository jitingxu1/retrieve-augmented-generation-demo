import logging
import time
import tempfile

import pinecone
from decouple import config
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from langchain import LLMChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from nltk.tokenize import sent_tokenize
from pinecone.core.client.model.vector import Vector
from langchain.chat_models import ChatOpenAI

from rag.schemas import QueryRequest, QueryResponse, InsertFileResponse
from rag.config import AppConfig
from pydantic import BaseModel

from rag.core.llm_chat.llm_chat import load_llm
from rag.core.embeddings.embedding_model import load_embedding
from rag.core.vector_store.base import load_vector_store

# ref: https://github.com/psychic-api/rag-stack/blob/main/server/server/main.py

load_dotenv()
router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("logfile.log"),  # Optionally, log to a file
    ]
)
PINECONE_API_KEY = config("PINECONE_API_KEY", cast=str)
PINECONE_ENVIRONMENT = config(
    "PINECONE_ENVIRONMENT",
    default="gcp-starter", cast=str
)
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
# index = pinecone.Index("rag-demo")
embeddings = load_embedding("openai")
vector_store = load_vector_store("pinecone", embeddings) # Pinecone(index, embeddings.embed_query, "text")
chat_model = load_llm("gpt-3.5-turbo")
MAX_CHUNK_SIZE = 1000
TEMPLATE = """
Answer question with above context using the following steps:
1) Context Relevance Check:
Verify whether the given context is relevant to the question being asked.

2) Answer Extraction from Relevant Context:
If the context is deemed relevant, summarize the answer in one sentence.

3) Handling Irrelevant Context:
In case the context is not pertinent to the question,
provide a response indicating "I do not know."

Context:
```{support_doc}```

Question:
{question}

"""

prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=["question", "support_doc"]
)

class ErrorMessage(BaseModel):
    detail: str

def nltk_senetnces(pdf: UploadFile = File(...)) -> list[str]:
    """
    Tokenize the sentences from the content of a PDF file.

    Args:
        pdf (UploadFile): The PDF file to process.

    Returns:
        list[str]: A list of sentences from the PDF content.
    """
    try:
        temp_file_path = tempfile.mktemp()
        with open(temp_file_path, "wb") as f:
            f.write(pdf.file.read())
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()
        page_contents = " ".join(
            [
                page.page_content.replace("\n", " ")
                for page in pages
            ]
        )
        return sent_tokenize(page_contents)
    except Exception as e:
        logger.error(f"error: {str(e)}")

def chunk_fixed_size(
    sentences: list[str],
    max_size: int
) -> list[str]:
    """
    Split a list of sentences into chunks with a maximum size.

    Args:
        sentences (list[str]): The list of sentences to be chunked.
        max_size (int): The maximum size of each chunk.

    Returns:
        list[str]: A list of chunks where each chunk
        contains sentences with a total length
        not exceeding max_size.
    """
    chunks = []
    a_chunk = []
    curr_size = 0
    for s in sentences:
        len_s = len(s)
        if curr_size + len_s <= max_size:
            a_chunk.append(s)
            curr_size += len_s
        else:
            chunks.append(" ".join(a_chunk))
            a_chunk = []
            curr_size = 0
    return chunks

@router.post(
    "/upload-files",
    response_model=InsertFileResponse,
    dependencies=[],
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorMessage,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorMessage,
        },
    },
)
def upload_files(
    pdf: UploadFile = File(...), # TODO: Consume generic files
    # configs: AppConfig() = None
) -> InsertFileResponse:
    try:
        sentences = nltk_senetnces(pdf)
        chunks = chunk_fixed_size(sentences, MAX_CHUNK_SIZE)
        docs = [Document(page_content=chunk) for chunk in chunks]
        ids = vector_store.add_documents(docs)
        logger.info(ids)
        return InsertFileResponse(success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    
@router.post(
    "/question",
    response_model=QueryResponse,
    # TODO: figure our how to do response, what does this mean
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorMessage,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorMessage,
        },
    },
    response_model_exclude_none=True,
)
def question(
    question: str, #QueryRequest,
) -> QueryResponse:
    try:
        # # Use high-level abstraction RetrieveQA in langchain
        # # https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch/blob/main/ask-titan-with-rag.py
        # # Also could implement our own retriever
        # # https://github.com/IntelLabs/fastRAG/blob/main/fastrag/retrievers/colbert.py
        documents = vector_store.similarity_search_with_score(
            query=question,
            k=2,
            filter=None,
            namespace=None
        )
        # # TODO: able to stwitch LLM for answering questions
        # # reference: https://github.com/zilliztech/akcio/tree/main/src_langchain/llm
        llm_chain = LLMChain(prompt=prompt, llm=chat_model)
        message = llm_chain.run(
            {
                "question": question,
                "support_doc": documents[0][0].page_content
            }
        )

        # TODO: Add callback functions to sync the QA to database
        return QueryResponse(
            query=question,
            documents=documents,
            answer=message,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))