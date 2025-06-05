import os
import logging
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_sqlserver.vectorstores import SQLServer_VectorStore


# Load environment variables
load_dotenv(override=True)

# Load document
print("Loading document...")
file_path = "../samples_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split text into chunks
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Initialize embedding model
print("Initializing embedding model...")
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
)

# Create vector store
print("Connecting to vector store...")
vector_store = SQLServer_VectorStore.from_documents(
    documents=all_splits,
    embedding=embedding_model,
    embedding_length=1536,
    connection_string=os.environ["MSSQL_CONNECTION_STRING"]
)

# Query the vectore store samples
print("Ready....")
print()

print("------------------------------------")
print("SAMPLE 01 - Querying vector store...")
print("------------------------------------")
print("Question: How many distribution centers does Nike have in the US?")
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print("Result:")
print(results[0])
print("\n\n")

print("-------------------------------------------")
print("SAMPLE 02 - Similarity search with score...")
print("-------------------------------------------")
print("Question: What was Nike's revenue in 2023?")
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}")
print("Result:")
print(doc)
print("\n\n")

print("--------------------------------------")
print("SAMPLE 03 - Using an embedded query...")
print("--------------------------------------")
print("Question: How were Nike's margins impacted in 2023?")
embedding = embedding_model.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print("Result:")
print(results[0])
print("\n\n")


print("-------------------------------")
print("SAMPLE 04 - Using retrievers...")
print("-------------------------------")
print("Question 1: How many distribution centers does Nike have in the US?")
print("Question 2: When was Nike incorporated?")

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print("Results:")
print(results)
print("\n\n")

print("Completed.")