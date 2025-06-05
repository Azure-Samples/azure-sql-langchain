import os
import logging

from dotenv import load_dotenv
from langchain_sqlserver.vectorstores import SQLServer_VectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

load_dotenv(override=True)

print(f"Setting up connection to Azure OpenAI embeddings ({os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]})...")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
)

print("Setting up connection to SQL Server...")
vector_store = SQLServer_VectorStore(
    connection_string=os.environ["MSSQL_CONNECTION_STRING"],
    table_name="langchain_test_table",
    embedding_function=embeddings,
    embedding_length=1536
)

print("Adding documents...")

def get_book(book_id):    
    with open(f"../samples_data/book{book_id}.txt", "r") as file:        
        payload = file.read()

    book = payload.split("---")
    return {            
            "title": book[0].strip(), 
            "author": book[1].strip(),
            "content": book[2].strip()
        }

for i in range(1, 4):
    print(f"Adding document {i}")        
    book = get_book(i)
    document = Document(page_content=book["content"], metadata={"book": book["title"], "author": book["author"]})
    vector_store.delete(ids=[str(i)])
    vector_store.add_documents([document], ids=[str(i)])
    
print("Delete sample document...")

vector_store.delete(ids=["4"])

query = "learn how to write T-SQL queries"
print()
print(f"Search: '{query}'")

print()
print(f"Simple search")
results = vector_store.similarity_search(query=query,k=2)
for doc in results:
    print(f"* {doc.metadata}")

print()
print("Search with filter")
results = vector_store.similarity_search(query=query,k=2,filter={"author": "Itzik Ben-Gan"}) 
for doc in results:
    print(f"* {doc.metadata}")

print()
print("Search with score")
results = vector_store.similarity_search_with_score(query=query,k=2)
for doc, score in results:
    print(f"* [Similarity Distance={score:3f}] {doc.metadata}")

print()
print("Use as Retriever")
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 2, "lambda_mult": 0.5},
)
results = retriever.invoke(query)
for doc in results:
    print(f"* {doc.metadata}")
