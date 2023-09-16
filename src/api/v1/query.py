import logging
import time

import pinecone
from decouple import config
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langchain import LLMChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone

from src.schemas import QueryRequest, QueryResponse

# ref: https://github.com/psychic-api/rag-stack/blob/main/server/server/main.py

load_dotenv()
router = APIRouter()
logger = logging.getLogger(__name__)
PINECONE_API_KEY = config("PINECONE_API_KEY", cast=str)
PINECONE_ENVIRONMENT = config(
    "PINECONE_ENVIRONMENT",
    default="gcp-starter", cast=str
)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index("langchain-demo")
embeddings = OpenAIEmbeddings()
vector_store = Pinecone(index, embeddings.embed_query, "text")
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


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[],
    responses={},
    response_model_exclude_none=True,
)
async def query(
    request: QueryRequest,
    config,  # add configureations
) -> QueryResponse:
    try:
        start_time = time.time()
        query = request.query
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
