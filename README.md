# LangChain samples with `langchain_sqlserver`

** Updated to use the `langchain_sqlserver` (0.1.2.) library. **

Get started with the [`langchain_sqlserver` library](https://github.com/langchain-ai/langchain-azure/tree/main/libs/sqlserver) with the following tutorials. All the tutorials works with Azure SQL or SQL Server 2025.

> [!NOTE]  
> SQL Server 2025 is available as Community Technology Preview (CTP) 1.0. To get more info on how to get the CTP, take a look here: [Announcing Microsoft SQL Server 2025: Enterprise AI-ready database from ground to cloud](https://www.microsoft.com/en-us/sql-server/blog/2024/11/19/announcing-microsoft-sql-server-2025-apply-for-the-preview-for-the-enterprise-ai-ready-database/)

> [!NOTE]  
> In Azure SQL, Vector Functions are in Public Preview. Learn the details about vectors in Azure SQL here: https://aka.ms/azure-sql-vector-public-preview

## Create an Azure SQL Database

If you don't have an Azure SQL Database yet, you can create one following the [quickstart guide](https://learn.microsoft.com/en-us/azure/azure-sql/database/single-database-create-quickstart?view=azuresql&tabs=azure-portal). Keep in mind that you can take advatange of the Azure SQL *free* tier: [Try Azure SQL Database for free](https://learn.microsoft.com/en-us/azure/azure-sql/database/free-offer?view=azuresql).

## LangChain Getting-Started Samples

If you're just getting started with LangChain, take a look at this article with the related sample: [LangChain Integration for Vector Support for SQL-based AI applications](https://devblogs.microsoft.com/azure-sql/langchain-with-sqlvectorstore-example/)

If you already have some familiarity with LangChain, and you are looking for samples that helps you to get started using LangChain with SQL Serve or Azure SQL, you can jump directly to the samples below.

All samples are availabel int the `./samples` folder:

```bash
cd samples
```

It is recommended that you use a virtual environment:

```bash
python -m venv .venv
```

And then make sure you have installed all required dependencies:

```bash
pip install -U -r .\requirements.txt
```

## LangChain Samples

Make sure the create an `.env` using `.env.example` as a template.

Samples on how to use the `langchain_sqlserver` library with SQL Server or Azure SQL as a vector store are:

- `test-1.py`: Basic sample to store vectors, content and metadata into SQL Server or Azure SQL and then do simple similarity searches.
- `test-2.py`: Read books reviews from a file, store it in SQL Server or Azure SQL, and then do similarity searches.

## LangChain Tutorials

### Build a semantic search engine

Build a semantic search engine over a PDF with document loaders, embedding models, and vector stores.

The tutorial described in the [Build a semantic search engine](https://python.langchain.com/docs/tutorials/retrievers/) page has been implemented in this project, but using the `langchain_sqlserver` library.

The file `semantic-search.py` contains the code of the tutorial. You can run it in your local environment. Make sure the create an `.env` using `.env.example` as a template.

The database used in the sample is named `langchain`. Make sure you have permission to create tables in the database.

### Build a Retrieval Augmented Generation (RAG) App: Part 1

Introduces RAG and walks through a minimal implementation, using LangGraph and LangChain.

The tutorial described in the [Build a Retrieval Augmented Generation (RAG) App: Part 1](https://python.langchain.com/docs/tutorials/rag/) page has been implemented in this project, but using the `langchain_sqlserver` library.

The file `rag-1.py` contains the code of the tutorial.

### Build a Retrieval Augmented Generation (RAG) App: Part 2

Extends the implementation to accommodate conversation-style interactions and multi-step retrieval processes, using LangGraph and LangChain.

The tutorial described in the [Build a Retrieval Augmented Generation (RAG) App: Part 2](https://python.langchain.com/docs/tutorials/qa_chat_history/) page has been implemented in this project, but using the `langchain_sqlserver` library.

The file `rag-2.py` contains the code of the tutorial.

### Agentic RAG

[Retrieval Agents](https://python.langchain.com/docs/tutorials/qa_chat_history/#agents) are useful when we want to make decisions about whether to retrieve data from an source.

The tutorial described in the [Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/) page has been implemented in this project, but using the `langchain_sqlserver` library.

The file `agentic-rag.py` contains the code of the tutorial.
