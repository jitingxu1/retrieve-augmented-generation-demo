
# Reference
- [llmsearch](https://github.com/snexus/llm-search/tree/main/src/llmsearch)
- [cobert](https://github.com/IntelLabs/fastRAG/blob/main/fastrag/retrievers/colbert.py)
- [verba](https://github.com/weaviate/Verba/blob/main/goldenverba/retrieval/advanced_engine.py)
- [rag](https://github.com/ray-project/llm-applications/blob/main/notebooks/rag.ipynb)
- [akcio](https://github.com/zilliztech/akcio/tree/main/src_towhee)
- [titan](https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch/blob/main/ask-titan-with-rag.py)
- [RAGstack](https://github.com/psychic-api/rag-stack/blob/main/server/server/main.py)
- [SEC-Insights](https://github.com/run-llama/sec-insights/blob/main/backend/app/api/crud.py)

# Retrieve Augmented Generation
Redegin the RAG system

## Setup instructions

### Install project dependencies

```
poetry install --with=dev
# anything else you'd like us to do to get the project running
```

### Copy and update the .env file

```
cp .env.example .env
# anything else you'd like us to do to get the project running
```

## Running

To run the service:

```
bin/dev
```

To run Integration test
```
./run_integration_test.sh
```

## Notes
- Best way to reduce hallucination is by retrieving userful and factual information
   1. Chunk size experimentation
   2. Chunk with contextual information: neighbors and parent information.
- Prompt emgineering
