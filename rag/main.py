import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from prometheus_client import start_http_server

from .api.router import router


def create_app() -> FastAPI:
    app = FastAPI(
        openapi_url="/openapi.json",
        # root_path=settings.API_ROOT_PATH,
        debug=False,
    )
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=["*"],
    #     allow_credentials=True,
    #     allow_methods=["*"],
    #     allow_headers=["*"],
    # )
    app.include_router(router)
    # init_logging()
    return app

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
logger.info("starting app...")
app = create_app()
app.include_router(router)

# METRICS_HTTP_SERVER_PORT = int(os.environ.get("METRICS_HTTP_SERVER_PORT", 8001))
# start_http_server(METRICS_HTTP_SERVER_PORT)

# # Set the features count on startup
# app.add_event_handler("startup", initialize_feature_count)