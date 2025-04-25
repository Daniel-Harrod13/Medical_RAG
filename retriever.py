# import Qdrant, sentence transformer, qdrantclient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

# bring embedings in from ingest.py

embeddings = SentenceTransformerEmbeddings(model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

# connect to qdrant client

url = "http://localhost:6333/dashboard"


client = QdrantClient(
    url = url,
    prefer_grpc= False,
)

 # print client
print(client) 

# set db

db = Qdrant(
    client = client,
    collection_name = "vector_database",
    embeddings = embeddings,
)

# print db
print(db)

print("########################")

# set query

query = "What are the common side effects of systemic theraputic agents?"

# set docs

docs = db.similarity_search_with_score(query=query, k = 2) 

for i in docs:
    doc, score = i
    print({"Score": score, "Content": doc.page_content, "metadata": doc.metadata})



