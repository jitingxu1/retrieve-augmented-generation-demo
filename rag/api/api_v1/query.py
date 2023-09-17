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
index = pinecone.Index("langchain-demo")
logger.info(index.__dict__)
embeddings = OpenAIEmbeddings()
logger.info(embeddings.openai_api_key)
vector_store = Pinecone(index, embeddings.aembed_query, "text")
gpt35 = OpenAI(model_name="gpt-3.5-turbo")
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
            "model": ErrorMessage, # custom pydantic model for 200 response
            "description": "Ok Response",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorMessage,  # custom pydantic model for 201 response
            "description": "Creates something from user request ",
        },
        status.HTTP_202_ACCEPTED: {
            "model": "",  # custom pydantic model for 202 response
            "description": "Accepts request and handles it later",
        },
    },
    response_model_exclude_none=True,
)
async def question(
    question: str,
):
    
    try:
        start_time = time.time()
        query = "hello"
        logger.info(f"question: {question}")
        logger.info(embeddings.openai_api_key)
        #logger.info(embeddings.embed_documents(texts=["Hello"]))
        question_embed = embeddings.embed_documents(texts=[question])
        logger.info(len(question_embed[0]))
        vector_store.add_texts(list(question))
        logger.info("inserted...")
        res = index.query(
            vector=question_embed,
            top_k=1
        )
        logger.info(f"*****{res}")
        # # Use high-level abstraction RetrieveQA in langchain
        # # https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch/blob/main/ask-titan-with-rag.py
        # # Also could implement our own retriever
        # # https://github.com/IntelLabs/fastRAG/blob/main/fastrag/retrievers/colbert.py
        # documents = await vector_store.similarity_search_with_score(
        #     query=query,
        #     k=1,
        #     filter=None,
        #     namespace=None
        # )
        # logger.info("query db")

        # # TODO: able to stwitch LLM for answering questions
        # # reference: https://github.com/zilliztech/akcio/tree/main/src_langchain/llm
        # llm_chain = LLMChain(prompt=prompt, llm=gpt35)
        # message = llm_chain.run(
        #     {
        #         "question": query,
        #         "support_doc": documents[0].page_content
        #     }
        # )
        # TODO: Add callback functions to sync the QA to database
        return QueryResponse(
            query=query,
            timings=f"{(time.time() - start_time):.2f}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/query",
    response_model=QueryResponse,
    #dependencies=[],
    responses={
        status.HTTP_200_OK: {
            "model": "", # custom pydantic model for 200 response
            "description": "Ok Response",
        },
        status.HTTP_201_CREATED: {
            "model": "",  # custom pydantic model for 201 response
            "description": "Creates something from user request ",
        },
        status.HTTP_202_ACCEPTED: {
            "model": "",  # custom pydantic model for 202 response
            "description": "Accepts request and handles it later",
        },
    },
    response_model_exclude_none=True,
)
async def query(
    # request: str, #QueryRequest,
    # config,  # add configureations
) -> QueryResponse:
    try:
        start_time = time.time()
        query = "hello"
        # Use high-level abstraction RetrieveQA in langchain
        # https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch/blob/main/ask-titan-with-rag.py
        # Also could implement our own retriever
        # https://github.com/IntelLabs/fastRAG/blob/main/fastrag/retrievers/colbert.py
        documents = await vector_store.similarity_search_with_score(
            query=query,
            k=1,
            filter=None,
            namespace=None
        )

        # TODO: able to stwitch LLM for answering questions
        # reference: https://github.com/zilliztech/akcio/tree/main/src_langchain/llm
        llm_chain = LLMChain(prompt=prompt, llm=gpt35)
        message = llm_chain.run(
            {
                "question": query,
                "support_doc": documents[0].page_content
            }
        )
        # TODO: Add callback functions to sync the QA to database
        return QueryResponse(
            query=query,
            answers=message,
            documents=documents,
            timings=f"{(time.time() - start_time):.2f}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
