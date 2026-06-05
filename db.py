# Silence the deprecation warnings from langchain packages
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Loading of the pdf
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader(r"document_loaders\deeplearning.pdf")
docs = loader.load()




# Chunking or splitting the text data
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,

)
chunks = splitter.split_documents(docs)


# Chunks to embeddings
from langchain_mistralai import MistralAIEmbeddings
embeddings = MistralAIEmbeddings(model="mistral-embed")


# Vector Database
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma-db"
)