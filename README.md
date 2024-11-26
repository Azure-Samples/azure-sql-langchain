# LangChain samples with `langchain_sqlserver`

Get started with the `langchain_sqlserver` library with the following tutorials. All the tutorials works with Azure SQL or SQL Server 2025.

> [!NOTE]  
> SQL Server 2025 is available as Community Technology Preview (CTP) 1.0. To get more info on how to get the CTP, take a look here: [Announcing Microsoft SQL Server 2025: Enterprise AI-ready database from ground to cloud](https://www.microsoft.com/en-us/sql-server/blog/2024/11/19/announcing-microsoft-sql-server-2025-apply-for-the-preview-for-the-enterprise-ai-ready-database/)

> [!NOTE]  
> In Azure SQL, Vector Functions are in Public Preview. Learn the details about vectors in Azure SQL here: https://aka.ms/azure-sql-vector-public-preview

## Build a semantic search engine

Build a semantic search engine over a PDF with document loaders, embedding models, and vector stores.

The tutorial described in the [Build a semantic search engine](https://python.langchain.com/docs/tutorials/retrievers/) page has been implemented in this project, but using the `langchain_sqlserver` library.

The file `./semantic-search.py` contains the code of the tutorial. You can run it in your local environment. Make sure the create an `.env` using `.env.example` as a template.

The database used in the sample is named `langchain`. Make sure you have permission to create tables in the database.

## Build a Retrieval Augmented Generation (RAG) App: Part 1

Introduces RAG and walks through a minimal implementation.

The tutorial described in the [Build a Retrieval Augmented Generation (RAG) App: Part 1](https://python.langchain.com/docs/tutorials/rag/) page has been implemented in this project, but using the `langchain_sqlserver` library.

The file `./rag-1.py` contains the code of the tutorial.

## Build a Retrieval Augmented Generation (RAG) App: Part 2

Extends the implementation to accommodate conversation-style interactions and multi-step retrieval processes.

The tutorial described in the [Build a Retrieval Augmented Generation (RAG) App: Part 2](https://python.langchain.com/docs/tutorials/qa_chat_history/) page has been implemented in this project, but using the `langchain_sqlserver` library.

The file `./rag-2.py` contains the code of the tutorial.

