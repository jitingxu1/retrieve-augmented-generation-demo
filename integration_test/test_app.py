import os
import time

import pytest
import requests
from fastapi.testclient import TestClient

from core.api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def reset_index():
    url = "http://127.0.0.1:8000/index_refresh/"
    response = requests.post(url)
    assert response.status_code == 200
    time.sleep(5)


def test_pdf_process(reset_index):
    url = "http://127.0.0.1:8000/process_pdf/"
    pdf_file = "./pdf/what-is-insurance_handout.pdf"
    files = {
        "pdf": open(pdf_file, "rb")
    }
    response = requests.post(url, files=files)
    data = response.json()

    assert response.status_code == 200
    assert data['cnt_new_llamma_vectors'] == 4
    assert data['cnt_new_openai_vectors'] == 4


def test_ask():
    api_url = "http://localhost:8000/ask"

    # Define the parameters for the request
    params = {
        "question": "what is insurance",
        "query_model": "llamma",
        "top_k": 1,
        "distance_threshold": 0,
        "pdf_document": "what-is-insurance_handout.pdf"
    }

    # Send a GET request to the API endpoint
    response = requests.get(api_url, params=params)
    assert response.status_code == 200
    data = response.json()
    assert len(data['answer']) > 1
