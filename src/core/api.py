import glob
import logging
import os
import tempfile
import time
import urllib.request
from typing import Annotated

import nltk
import openai
import pinecone
from decouple import config
from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, UploadFile
from langchain import LLMChain, PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp, OpenAI
from nltk.tokenize import sent_tokenize
from pinecone.index import Index

nltk.download('punkt')

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


LLM_CONTEXT_SIZE_IN_TOKENS = config(
    "LLM_CONTEXT_SIZE_IN_TOKENS",
    default=4096,
    cast=int
)
MAX_CHUNK_SIZE_LLAMA = config("MAX_CHUNK_SIZE", default=1024, cast=int)
MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING = config(
    "MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING",
    default=4,
    cast=int
)
DEFAULT_MODEL_NAME = config(
    "DEFAULT_MODEL_NAME",
    default="llama2_7b_chat_uncensored", cast=str
)
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MIN_CONTEXT_SIZE = 5
LLAMMA_INDEX_NAME = "llamma-index"
LLAMA_EMD_DIM = 4096
LLM_MODEL_URL = config("LLM_MODEL_URL", cast=str)
OPENAI_INDEX_NAME = "openai-index"
OPENAI_ADA_EMD_DIM = 1536
OPENAI_EMBEDDINGS_MODEL = "text-embedding-ada-002"
PINECONE_ENVIRONMENT = config(
    "PINECONE_ENVIRONMENT",
    default="gcp-starter", cast=str
)
PINECONE_ENVIRONMENT_2 = config(
    "PINECONE_ENVIRONMENT_2",
    default="us-west1-gcp-free", cast=str
)
GPT_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = config("OPENAI_API_KEY", cast=str)
PINECONE_API_KEY = config("PINECONE_API_KEY", cast=str)
PINECONE_API_KEY_2 = config("PINECONE_API_KEY_2", cast=str)

embedding_model_cache = {}
llamma_embedings_cache = {}
openai_embedings_cache = {}

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

load_dotenv()
app = FastAPI()


def download_model(url: str) -> str:
    """
    Download a model file from the specified URL.

    Args:
        url (str): The URL from which to download the model.

    Returns:
        str: The name of the downloaded model file.
    """
    logger.info("Starting download models...")
    model_name = os.path.basename(url)
    logger.info(f"model_name = {model_name}")
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    filename = os.path.join(models_dir, model_name)
    if not os.path.exists(filename):
        logger.info(f"Downloading model {model_name} from {url}...")
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Downloaded: {filename}")
    else:
        logger.info(f"File already exists: {filename}")
    return model_name


def load_model(model_name: str) -> tuple[LlamaCppEmbeddings, LlamaCpp]:
    """
    Load an embedding model instance and a language model instance.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple[LlamaCppEmbeddings, LlamaCpp]:
            A tuple containing the embedding model
            and language model instances.
    """
    models_dir = os.path.join(BASE_DIRECTORY, 'models')
    if model_name in embedding_model_cache:
        return embedding_model_cache[model_name]
    matching_files = glob.glob(os.path.join(models_dir, f"{model_name}*"))
    if not matching_files:
        logger.error(f"No model file found matching: {model_name}")
        raise FileNotFoundError
    matching_files.sort(key=os.path.getmtime, reverse=True)
    model_file_path = matching_files[0]
    model_instance = LlamaCppEmbeddings(
        model_path=model_file_path,
        use_mlock=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS
    )
    llm = LlamaCpp(model_path=model_file_path, temperature=0.75, top_p=1)
    model_instance.client.verbose = False
    embedding_model_cache[model_name] = model_instance
    return model_instance, llm


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


def check_exist_index(
    id: str,
    pinecone_index: Index
) -> bool:
    """
    Check if a vector with the given ID exists in a Pinecone index.

    Args:
        id (str): The ID of the vector to check.
        pinecone_index (pinecone.index.Index): The Pinecone index to search in.

    Returns:
        bool: True if the vector with the given ID exists
        in the index, False otherwise.
    """
    return id in pinecone_index.fetch(ids=[id]).vectors


def get_llm_embeddings(
    text: str,
    text_hash: str,
    file_name: str
) -> dict:
    """
    Calculate and retrieve LlaMMa embeddings for the given text.

    Args:
        text (str): The text for which embeddings need to be calculated.
        text_hash (str): A unique identifier (hash) for the text.
        file_name (str): The name of the associated file.

    Returns:
        dict: A dictionary containing the calculated embeddings
        and associated metadata.
    """
    if text_hash in llamma_embedings_cache:
        logger.warn("Loading LlaMMa Embedding from Cache..")
        embed = llamma_embedings_cache[text_hash]
    else:
        logger.warn("Llamma is calculating vectors...")
        embed = llm_embed.embed_query(text)
        llamma_embedings_cache[text_hash] = embed
    return {
        "id": str(text_hash),
        'values': embed,
        'metadata': {
            "text": text,
            "file_name": file_name,
        },
    }


def get_openai_embeddings(
    text: str,
    text_hash: str,
    file_name: str
) -> dict:
    """
    Calculate and retrieve OpenAI embeddings for the given text.

    Args:
        text (str): The text for which embeddings need to be calculated.
        text_hash (str): A unique identifier (hash) for the text.
        file_name (str): The name of the associated file.

    Returns:
        dict: A dictionary containing the calculated embeddings
        and associated metadata.
    """
    if text_hash in openai_embedings_cache:
        logger.warn("Loading OpenAI Embedding from Cache..")
        embed = openai_embedings_cache[text_hash]
    else:
        res = openai.Embedding.create(
            input=[text],
            engine=OPENAI_EMBEDDINGS_MODEL
        )
        embed = res['data'][0]['embedding']
        logger.info(f"Open AI embedding length is {len(embed)}")
        openai_embedings_cache[text_hash] = embed
    return {
        "id": str(text_hash),
        "values": embed,
        "metadata": {
            "text": text,
            "file_name": file_name,
        },
    }


def generate_support_docs(
    llm_index: Index,
    vector: list[float],
    pdf_document: str,
    top_k: int = 3,
    distance_threshold: float = 0.0
) -> str:
    """
    Generate support documents by querying the LLM index with a given vector.

    Args:
        llm_index (Index): The LLM index to query.
        vector (list[float]): The vector used for querying.
        top_k (int): The number of top matches to retrieve.
        pdf_document (str): The PDF document to filter by.
        distance_threshold (float): The threshold for matching scores.

    Returns:
        str: A formatted string containing the matched context documents.
    """
    returned = llm_index.query(
        top_k=int(top_k),
        filter={
            "file_name": {"$eq": pdf_document}
        },
        include_values=True,
        include_metadata=True,
        vector=vector
    )
    if len(returned.matches) < 1:
        return ""
    logger.info(f"Matched {len(returned.matches)}")
    logger.info(f"Max score is {returned.matches[0].score}")
    logger.info(f"Min score is {returned.matches[-1].score}")
    context = [m.metadata['text']
               for m in returned.matches if m.score > distance_threshold]
    return "\n\n".join(context)


@app.get("/hello")
def hello(name: Annotated[str, Query()] = "Wilson"):
    return {"message": f"Hello {name}!"}


@app.post("/index_refresh/")
async def refresh_index():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    if LLAMMA_INDEX_NAME in pinecone.list_indexes():
        logger.critical(f"Delete index: {LLAMMA_INDEX_NAME}...")
        pinecone.delete_index(LLAMMA_INDEX_NAME)
    logger.info(f"creating index: {LLAMMA_INDEX_NAME}")
    pinecone.create_index(
        LLAMMA_INDEX_NAME,
        dimension=LLAMA_EMD_DIM,
        metric='cosine'
    )
    pinecone.init(api_key=PINECONE_API_KEY_2,
                  environment=PINECONE_ENVIRONMENT_2)
    if OPENAI_INDEX_NAME in pinecone.list_indexes():
        logger.critical(f"Delete index: {OPENAI_INDEX_NAME}")
        pinecone.delete_index(OPENAI_INDEX_NAME)
    logger.info(f"create openai index: {OPENAI_INDEX_NAME}")
    pinecone.create_index(
        OPENAI_INDEX_NAME,
        dimension=OPENAI_ADA_EMD_DIM,
        metric='cosine'
    )
    time.sleep(1)
    return {
        "message": "success"
    }


@app.post("/process_pdf/")
async def upload_pdf(
    pdf: UploadFile = File(...),
):
    """
    Process an uploaded PDF file by tokenizing sentences,
    calculating embeddings, and storing them in Pinecone indexes.

    Args:
        pdf (UploadFile): The PDF file to process.

    Returns:
        dict: A dictionary containing information about the processed PDF.
    """
    # Convert PDF contents into cleaned sentence format
    # Using PYPDF and NLTK
    sentences = nltk_senetnces(pdf)
    # Merge sentences into chunks with a max size limit
    chunks = chunk_fixed_size(sentences, MAX_CHUNK_SIZE_LLAMA)

    llamma_sentences_embedings = []
    openai_sentences_embedings = []

    # Cacculate embeddings
    for index, text in enumerate(chunks):
        logger.info(f"Chunk {index}'s size is {len(text)}")
        text_hash = str(hash(text))
        logger.info(f"embedding id = {text_hash}")
        # Pull LlaMma embeddings
        logger.info("Ingest PDF document using llamma embedings..")
        if not check_exist_index(text_hash, llamma_index):
            llamma_sentences_embedings.append(
                get_llm_embeddings(text, text_hash, pdf.filename)
            )
        # Pull OpenAI embeddings
        logger.info("Process document by openai embeddings...")
        if not check_exist_index(text_hash, openai_index):
            openai_sentences_embedings.append(
                get_openai_embeddings(text, text_hash, pdf.filename)
            )

    # Load vectors to pinecone index:
    if len(openai_sentences_embedings) > 0:
        openai_index.upsert(vectors=openai_sentences_embedings)
    if len(llamma_sentences_embedings) > 0:
        llamma_index.upsert(vectors=llamma_sentences_embedings)

    llamma_stats = llamma_index.describe_index_stats()
    openai_stats = openai_index.describe_index_stats()

    return {
        "cnt_new_openai_vectors": len(openai_sentences_embedings),
        "cnt_new_llamma_vectors": len(llamma_sentences_embedings),
        "total_openai_vector_count": openai_stats.total_vector_count,
        "total_llamma_vector_count": llamma_stats.total_vector_count,
        "openai_dimension": openai_stats.dimension,
        "llamma_dimension": llamma_stats.dimension,
        "file_name": pdf.filename,
        "openai_index_fullness": openai_stats.index_fullness,
    }


@app.get("/ask")
def query(
    question: str,
    query_model: str = "llamma",
    top_k: int = 3,
    distance_threshold: float = 0,
    pdf_document: str = None
):
    """
    Perform a question-answering query using specified models
    and a support document.

    Args:
        question (str): The question to ask.
        query_model (str): The query model to use ("llamma" or "openai").
        top_k (int): The number of top matches to consider.
        distance_threshold (float): The distance threshold for matching scores.
        pdf_document (str): The PDF document to use as support.

    Returns:
        dict: A dictionary containing the question, context, and answer.
    """
    logger.info(f"Question: {question}")
    logger.info(f"Model: {query_model}")
    logger.info(f"pdf_document: {pdf_document}")
    logger.info(f"top_k = {top_k}")
    prompt = PromptTemplate(
        template=TEMPLATE,
        input_variables=["question", "support_doc"]
    )
    support_doc = ""
    message = ""
    if query_model == "llamma":
        logger.info(f"Ask LlaMma with {pdf_document}")
        q_embed = llm_embed.embed_query(question)
        support_doc = generate_support_docs(
            llamma_index,
            q_embed,
            pdf_document,
            top_k,
            distance_threshold
        )
        if len(support_doc) <= MIN_CONTEXT_SIZE:
            logger.info(f"No context found in {pdf_document}")
            return {
                "question": question,
                "context": "No matches",
                "answer": "No answer",
            }
        logger.info(f"LlaMMa Context: {support_doc}")
        llm_chain = LLMChain(prompt=prompt, llm=llm_chat)
        message = llm_chain.run(
            {
                "question": question,
                "support_doc": support_doc
            }
        )
    elif query_model == "openai":
        logger.info("Ask GPT3.5...")
        q_embed = openai.Embedding.create(
            input=[question],
            engine=OPENAI_EMBEDDINGS_MODEL
        )['data'][0]['embedding']
        support_doc = generate_support_docs(
            openai_index,
            q_embed,
            pdf_document,
            top_k,
            distance_threshold
        )
        logger.info(f"OpenAI Context:\n{support_doc}")
        if len(support_doc) < MIN_CONTEXT_SIZE:
            logger.info(f"No context found in {pdf_document}")
            return {
                "question": question,
                "context": "No matches",
                "answer": "No answer",
            }

        gpt35_chain = LLMChain(prompt=prompt, llm=gpt35)
        message = gpt35_chain.run(
            {
                "question": question,
                "support_doc": support_doc
            }
        )

    return {
        "question": question,
        "context": support_doc,
        "answer": message,
    }


@app.on_event("startup")
async def startup_event():
    """
    Initialize Pinecone indexes and models upon startup.
    """
    global llamma_index, openai_index, llm_embed, llm_chat, gpt35

    # Load llamma ccp
    model_name = download_model(LLM_MODEL_URL)
    llm_embed, llm_chat = load_model(model_name)

    # setup open source llm and pinecone index
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    if LLAMMA_INDEX_NAME not in pinecone.list_indexes():
        logger.info(f"creating index: {LLAMMA_INDEX_NAME}")
        pinecone.create_index(
            LLAMMA_INDEX_NAME,
            dimension=LLAMA_EMD_DIM,
            metric='cosine'
        )
        time.sleep(1)
    llamma_index = pinecone.Index(LLAMMA_INDEX_NAME)

    # setup open ai and pinecone index
    gpt35 = OpenAI(openai_api_key=OPENAI_API_KEY)
    openai.api_key = OPENAI_API_KEY
    pinecone.init(api_key=PINECONE_API_KEY_2,
                  environment=PINECONE_ENVIRONMENT_2)
    if OPENAI_INDEX_NAME not in pinecone.list_indexes():
        logger.info(f"create openai index: {OPENAI_INDEX_NAME}")
        pinecone.create_index(
            OPENAI_INDEX_NAME,
            dimension=OPENAI_ADA_EMD_DIM,
            metric='cosine'
        )
        time.sleep(1)
    openai_index = pinecone.Index(OPENAI_INDEX_NAME)

    logger.info(f"Loaded llamma index: {LLAMMA_INDEX_NAME}")
    logger.info(llamma_index.describe_index_stats())
    logger.info(f"Loaded OpenAI index: {OPENAI_INDEX_NAME}")
    logger.info(openai_index.describe_index_stats())
