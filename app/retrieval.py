import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import dotenv
from langchain_chroma import Chroma
from langchain.chat_models import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough #Takes an input and returns it unchanged
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


# Load environment variables from .env file
dotenv.load_dotenv()

memory = ConversationBufferMemory(memory_key="history", return_messages=True)
persist_directory = "./chroma_langchain_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)

llm = GoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.1,
    max_output_tokens=1024
)

health_prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
    You are a helpful health assistant. 
    Use the retrieved context to answer the user’s question. 
    If you are unsure, say so and recommend consulting a healthcare professional.

    Conversation so far:
    {history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

qa_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | health_prompt
    | llm
    | StrOutputParser()
)

qa_chain.invoke("What are autonomous agents?")


def ask_health_agent(query: str):
    response = qa_chain.invoke(query)
    memory.save_context({"input": query}, {"output": response})
    return response
