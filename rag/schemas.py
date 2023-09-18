from typing import Dict, List, Optional, Tuple

# from haystack.schema import Answer, Document
from pydantic import BaseConfig, BaseModel, Extra, Field

from langchain.schema.document import Document


class BaseRequest(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class QueryRequest(BaseRequest):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False

class BaseResponse(BaseModel):
    class Config:
        ...


class QueryResponse(BaseModel):
    query: str
    documents: List[Tuple[Document, float]]
    answer: str

class InsertFileResponse(BaseResponse):
    success: bool
