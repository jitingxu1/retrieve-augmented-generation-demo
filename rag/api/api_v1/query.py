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
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
# logger.info(PINECONE_API_KEY, PINECONE_ENVIRONMENT)
index = pinecone.Index("rag-demo")
logger.info(index.__dict__)
embeddings = OpenAIEmbeddings()
logger.info(embeddings.openai_api_key)
vector_store = Pinecone(index, embeddings.embed_query, "text")
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo")
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

@router.post(
    "/upload-files",
    response_model=InsertFileResponse,
    dependencies=[],
    # response={},
)
async def upload_files(
    pdf: UploadFile = File(...), # TODO: Consume generic files
    # configs: AppConfig() = None
) -> InsertFileResponse:
    try:
        logger.log("Upload files..")
        sentences = nltk_senetnces(pdf)
        doc = Document(page_content=" ".join(sentences))
        success = await vector_store.aadd_documents(doc)
        return InsertFileResponse(success=success)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class ErrorMessage(BaseModel):
    detail: str
    
@router.post(
    "/question",
    response_model=QueryResponse,
    # TODO: figure our how to do response, what does this mean
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorMessage,
            "description": "Ok Response",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorMessage,
            "description": "Creates something from user request ",
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
        llm_chain = LLMChain(prompt=prompt, llm=gpt35)
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