# import libraries 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings

# Create embedding variables to hold the model  
embeddings = SentenceTransformerEmbeddings(model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

print(embeddings)

# build the directory loader 

loader = DirectoryLoader('Data/', glob= '*.pdf', show_progress=True, loader_cls=PyPDFLoader)

documents = loader.load()

#build text splitter recursive
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
) 

texts = text_splitter.split_documents(documents)

url = "http://localhost:6333/dashboard"

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url = url,
    prefer_grpc= False,
    collection_name= "vector_database",

)

# print Vector Database is created
print("Vector Database is created!")













