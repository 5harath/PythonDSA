from openai import OpenAI
from pymilvus import MilvusClient

MODEL_NAME = "text-embedding-ada-002"  # Which model to use, please check https://platform.openai.com/docs/guides/embeddings for available models
DIMENSION = 1536  # Dimension of vector embedding

# Connect to OpenAI with API Key.
openai_client = OpenAI(api_key="")

# Read data from source.txt and assign it to the docs variable.
with open("source.txt", "r", encoding="utf-8") as file:
    docs = [line.strip() for line in file if line.strip()]

# Generate embeddings using OpenAI API.
vectors = [
    vec.embedding
    for vec in openai_client.embeddings.create(input=docs, model=MODEL_NAME).data
]

# Prepare data to be stored in Milvus vector database.
# We can store the id, vector representation, raw text and labels such as "subject" in this case in Milvus.
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(docs))
]

print(data)

# Connect to Milvus, all data is stored in a local file named "milvus_openai_demo.db"
# in current directory. You can also connect to a remote Milvus server following this
# instruction: https://milvus.io/docs/install_standalone-docker.md.
milvus_client = MilvusClient(uri="", token="")
COLLECTION_NAME = "demo_collection1"  # Milvus collection name
# Create a collection to store the vectors and text.
if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    milvus_client.drop_collection(collection_name=COLLECTION_NAME)
milvus_client.create_collection(collection_name=COLLECTION_NAME, dimension=DIMENSION)

# Insert all data into Milvus vector database.
res = milvus_client.insert(collection_name="demo_collection_copy", data=data)

print(res["insert_count"])