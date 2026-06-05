import warnings
# Silence the deprecation warnings from langchain packages
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()


# Importing essential libraries
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Turning a user querry to embeddings
embedding_model = MistralAIEmbeddings()

# Mentioning the vector database
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding=embedding_model
)


# Creating retrieval
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs ={
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5 
    }
)

# Creating LLM
llm = ChatMistralAI(model = "mistral-small-2506")


# Creating prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)





