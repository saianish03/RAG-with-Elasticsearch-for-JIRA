# RAG for JIRA Data with Elasticsearch

This project implements a **Retrieval Augmented Generation (RAG)** system to analyze historical JIRA bug reports, leveraging **LlamaIndex** and **Elasticsearch** to process and index semantic embeddings from Apache Buildr project data.

## Overview
This project processes bug-fix reports over a 10-year period from Apache's open-source projects, sourced from the [PROMISE'19 Dataset](https://figshare.com/articles/dataset/Replication_Package_-_PROMISE_19/8852084). The system is built to:
- Index and process 50+ fields per issue.
- Implement efficient data transformation for optimal context preservation.
- Create semantic embeddings using **OllamaEmbedding** using local Ollama (StableLM-2 model).
- Store these semantic embeddings in **Elasticsearch** VectorStore
- Perform natural language queries over bug descriptions, comments, and commits.