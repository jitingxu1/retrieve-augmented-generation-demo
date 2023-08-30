from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from core.api import app


@pytest.fixture
def mock_download_model():
    with patch('core.api.download_model',
               return_value='mocked_model_name'
               ):
        yield mock_download_model


@pytest.fixture
def mock_load_model():
    mock_llm_embed = Mock()
    mock_llm_chat = Mock()
    with patch('core.api.load_model',
               return_value=(mock_llm_embed, mock_llm_chat)
               ) as mock_load_model:
        yield mock_load_model


def test_hello(
    mock_download_model,
    mock_load_model,
):
    with TestClient(app) as client:
        response = client.get("/hello?name=Reba")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello Reba!"}
        assert client.get("/hello").json() == {"message": "Hello Wilson!"}


@patch('core.api.download_model', return_value='mocked_model_name')
@patch('core.api.load_model', return_value=('mocked_model_name', "kk"))
@patch('core.api.chunk_fixed_size', return_value=["s1"])
@patch('core.api.get_llm_embeddings',
       return_value={"id": 1, "value": [0.1, 0.2]}
       )
@patch('core.api.get_openai_embeddings',
       return_value={"id": 1, "value": [0.0, 0.0]}
       )
@patch('core.api.check_exist_index', return_value=False)
def test_process_pdf(
    mock_download_model,
    mock_load_model,
    mock_chunk_fixed_size,
    mock_get_llm_embeddings,
    mock_get_openai_embeddings,
    mock_check_exist_index,
):
    mock_llamma_index = Mock()
    mock_openai_index = Mock()
    mock_llamma_stats = mock_llamma_index.describe_index_stats.return_value
    mock_llamma_stats.total_vector_count = 2
    mock_llamma_stats.dimension = 2

    # Mock the return values for openai_index.describe_index_stats()
    mock_openai_stats = mock_openai_index.describe_index_stats.return_value
    mock_openai_stats.total_vector_count = 2
    mock_openai_stats.dimension = 2
    mock_openai_stats.index_fullness = 0.01
    with patch("core.api.openai_index", mock_openai_index), \
            patch("core.api.llamma_index", mock_llamma_index):
        client = TestClient(app)
        response = client.post(
            "/process_pdf/",
            files={"pdf": ("dummy.pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "cnt_new_openai_vectors" in data
        assert data["cnt_new_openai_vectors"] == 1
        assert data["cnt_new_llamma_vectors"] == 1
        assert data["openai_dimension"] == 2
        assert mock_get_llm_embeddings.call_count == 1


@patch("core.api.generate_support_docs", return_value="Hi")
def test_ask(
    mock_generate_support_docs,
):
    with TestClient(app) as client:
        response = client.get(
            "/ask",
            params={
                "question": "What is the meaning of life?",
                "query_model": "llamma",
                "pdf_document": "document.pdf",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "context" in data
        assert "answer" in data
