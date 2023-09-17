from fastapi import APIRouter

from .api_v1.query import router as query_router
from .api_v1.root import router as root_router

router = APIRouter()
router.include_router(query_router)
router.include_router(root_router)