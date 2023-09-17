from typing import Dict, List, Optional

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
    answers: Optional[List] = []
    documents: List[Document] = []
    images: Optional[Dict] = None
    relations: Optional[List] = None
    debug: Optional[Dict] = Field(None, alias="_debug")
    timings: Optional[Dict] = None
    results: Optional[List] = None

class InsertFileResponse(BaseResponse):
    success: bool
