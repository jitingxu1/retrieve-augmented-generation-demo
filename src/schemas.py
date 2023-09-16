from pydantic import BaseConfig, BaseModel, Extra, Field
from typing import Optional, List, Dict
from haystack.schema import Answer, Document


class BaseRequest(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid

class QueryRequest(BaseRequest):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False

class QueryResponse(BaseModel):
    query: str
    answers: Optional[List] = []
    documents: List[Document] = []
    images: Optional[Dict] = None
    relations: Optional[List] = None
    debug: Optional[Dict] = Field(None, alias="_debug")
    timings: Optional[Dict] = None
    results: Optional[List] = None
