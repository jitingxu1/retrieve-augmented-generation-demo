# Reference

- [llmsearch](https://github.com/snexus/llm-search/tree/main/src/llmsearch)
- [cobert](https://github.com/IntelLabs/fastRAG/blob/main/fastrag/retrievers/colbert.py)
- [verba](https://github.com/weaviate/Verba/blob/main/goldenverba/retrieval/advanced_engine.py)
- [rag](https://github.com/ray-project/llm-applications/blob/main/notebooks/rag.ipynb)
- [akcio](https://github.com/zilliztech/akcio/tree/main/src_towhee)
- [titan](https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch/blob/main/ask-titan-with-rag.py)
- [RAGstack](https://github.com/psychic-api/rag-stack/blob/main/server/server/main.py)
- [SEC-Insights](https://github.com/run-llama/sec-insights/blob/main/backend/app/api/crud.py)
- [gpt4-pdf-chatbot-langchain](https://github.com/mayooear/gpt4-pdf-chatbot-langchain)
- [frontend repo](https://github.com/zahidkhawaja/langchain-chat-nextjs)

# Learnings: 
- [reference: llm-search](https://www.reddit.com/r/LocalLLaMA/comments/16cbimi/yet_another_rag_system_implementation_details_and/)
## Pre-processing and chunking

- Document format - the best quality is achieved with a format where the logical structure of the document can be parsed - titles, headers/subheaders, tables, etc. Examples of such formats include markdown, HTML, or .docx.

- PDFs, in general, are hard to parse due to multiple ways to represent the internal structure - for example, it can be just a bunch of images stacked together. In most cases, expect to be able to split by sentences.

- Content splitting:

  1. Splitting by logical blocks (e.g., headers/subheaders) improved the quality significantly. It comes at the cost of format-dependent logic that needs to be implemented. Another downside is that it is hard to maintain an equal chunk size with this approach.

  2. For documents containing source code, it is best to treat the code as a single logical block. If you need to split the code in the middle, make sure to embed metadata providing a hint that different pieces of code are related.

  3. Metadata included in the text chunks:

      - Document name.

      - References to higher-level logical blocks (e.g., pointing to the parent header from a subheader in a markdown document).

      - For text chunks containing source code - indicating the start and end of the code block and optionally the name of the programming language.

  4. External metadata - added as external metadata in the vector store. These fields will allow dynamic filtering by chunk size and/or label.

      - Chunk size.

      - Document path.

      - Document collection label, if applicable.

  5. Chunk sizes - as many people mentioned, there appears to be high sensitivity to the chunk size. There is no universal chunk size that will achieve the best result, as it depends on the type of content, how generic/precise the question asked is, etc.

      - One of the solutions is embedding the documents using multiple chunk sizes and storing them in the same collection.

      - During runtime, querying against these chunk sizes and selecting dynamically the size that achieves the best score according to some metric.

      - Downside - increases the storage and processing time requirements.



## Embeddings

- There are multiple embedding models achieving the same or better quality as OpenAI's ADA - for example, `e5-large-v2` - it provides a good balance between size and quality.

- Some embedding models require certain prefixes to be added to the text chunks AND the query - that's the way they were trained and presumably achieve better results compared to not appending these prefixes.



## Retrieval

- One of the main components that allowed me to improve retrieval is a **re-ranker**. A re-ranker allows scoring the text passages obtained from a similarity (or hybrid) search against the query and obtaining a numerical score indicating how relevant the text passage is to the query. Architecturally, it is different (and much slower) than a similarity search but is supposed to be more accurate. The results can then be sorted by the numerical score from the re-ranker before stuffing into LLM.

- A re-ranker can be costly (time-consuming and/or require API calls) to implement using LLMs but is efficient using cross-encoders. It is still slower, though, than cosine similarity search and can't replace it.

- Sparse embeddings - I took the general idea from [Getting Started with Hybrid Search | Pinecone](https://www.pinecone.io/learn/hybrid-search-intro/) and implemented sparse embeddings using SPLADE. This particular method has an advantage that it can minimize the "vocabulary mismatch problem." Despite having large dimensionality (32k for SPLADE), sparse embeddings can be stored and loaded efficiently from disk using Numpy's sparse matrices.

- With sparse embeddings implemented, the next logical step is to use a **hybrid search** - a combination of sparse and dense embeddings to improve the quality of the search.

- Instead of following the method suggested in the blog (which is a weighted combination of sparse and dense embeddings), I followed a slightly different approach:

  1. Retrieve the **top k** documents using SPLADE (sparse embeddings).

  2. Retrieve **top k** documents using similarity search (dense embeddings).

  3. Create a union of documents from sparse or dense embeddings. Usually, there is some overlap between them, so the number of documents is almost always smaller than 2*k.

  4. Re-rank all the documents (sparse + dense) using the re-ranker mentioned above.

  5. Stuff the top documents sorted by the re-ranker score into the LLM as the most relevant documents.

  6. The justification behind this approach is that it is hard to compare the scores from sparse and dense embeddings directly (as suggested in the blog - they rely on magical weighting constants) - but the re-ranker should explicitly be able to identify which document is more relevant to the query.

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

- Best way to reduce hallucination is by retrieving useful and factual information
  1.  Chunk size experimentation
  2.  Chunk with contextual information: neighbors and parent information.
- Prompt emgineering
