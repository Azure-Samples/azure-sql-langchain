import os
import logging

from dotenv import load_dotenv
from langchain_sqlserver.vectorstores import SQLServer_VectorStore
from langchain_openai import AzureOpenAIEmbeddings

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

load_dotenv() 

print(f"Setting up connection to Azure OpenAI embeddings ({os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]})...")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
)

connection_string = os.environ["MSSQL_CONNECTION_STRING"]

print("Setting up connection to SQL Server table 1...")

store1 = SQLServer_VectorStore(
    connection_string=connection_string,
    embedding_function=embeddings,
    table_name="langchain_test_table1",
    embedding_length=1536
)

print("Setting up connection to SQL Server table 2...")

store2 = SQLServer_VectorStore(
    connection_string=connection_string,
    embedding_function=embeddings,
    table_name="langchain_test_table2",
    embedding_length=1536
)

print("Creating payload...")

payload = [
    {'id': 1, 'text': "Apples and oranges"},
    {'id': 2, 'text': "Cars and airplanes"},
    {'id': 3, 'text': "Pineapple" },
    {'id': 4, 'text': "Train"},
    {'id': 5, 'text': "Bananas"},
    {'id': 6, 'text': "Boats"},    
    {'id': 7, 'text': "Vessels"}    
]

ids = [p['id'] for p in payload]
texts = [p['text'] for p in payload]
metadatas = [{'len': len(p['text'])} for p in payload]

print("Delete text in table 1...")

store1.delete(ids)

print("Storing text in table 1...")

store1.add_texts(
    ids=ids,
    texts=texts,
    metadatas=metadatas
)

print("Delete text in table 2...")

store2.delete(ids)

print("Storing text in table 2...")

store2.add_texts(
    ids=ids,
    texts=texts,
    metadatas=metadatas
)

print()
search_text = "ships"
print(f"Similarity search for '{search_text}' in table 1...")
result1 = store1.similarity_search(search_text, 3)
print(result1)

print()
search_text = "ships"
print(f"Similarity search for '{search_text}' with score in table 1...")
result1 = store1.similarity_search_with_score(search_text, 3)
print(result1)

print()
search_text = "ships"
print(f"Similarity search for '{search_text}' as retriever with score in table 1...")
retriever = store1.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}
)
result1 = retriever.invoke(search_text)
print(result1)

print()
print("Done.")


