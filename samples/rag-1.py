import os
import logging
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
load_dotenv(override=True)

import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_sqlserver.vectorstores import SQLServer_VectorStore

from langgraph.graph import START, StateGraph

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# Initialize the Azure Chat OpenAI model.
print("Initializing Azure Chat OpenAI...")
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_CHAT_API_VERSION"],
)

# Load, chunk and index the contents of the blog.
print("Loading sample blog post...")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

print("Splitting blog post...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

print("Saving chunks into vector store...")
vector_store = SQLServer_VectorStore.from_documents(
    documents=all_splits, 
    embedding=AzureOpenAIEmbeddings(azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]),
    embedding_length=1536,
    connection_string=os.environ["MSSQL_CONNECTION_STRING"],
    table_name="rag1"
)

print("Initializing RAG pattern...")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.            
            Context: {context} 
            """,
        ),
        (
            "human",
            "Question: {question} ?",
        ),
    ]
)

# Define the state class
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define the retrieve function
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Define the generate function
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build the graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

## Invoke the graph 
print("Invoking the graph...")
print("Question: What is Task Decomposition?")
result = graph.invoke({"question": "What is Task Decomposition?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')