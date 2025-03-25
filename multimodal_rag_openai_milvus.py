from openai import OpenAI
from pymilvus import MilvusClient
import pandas as pd

# OpenAI and Milvus configuration
MODEL_NAME = "text-embedding-ada-002"
DIMENSION = 1536
COLLECTION_NAME = "demo_collection_copy"

# Initialize OpenAI client
openai_client = OpenAI(api_key="")

# Initialize Milvus client
milvus_client = MilvusClient(uri="", token="")

# Function to generate embeddings using OpenAI
def generate_embedding(text):
    return openai_client.embeddings.create(input=[text], model=MODEL_NAME).data[0].embedding

# Function to search similar text in Milvus
def search_similar_text(query_text, top_n=3):
    query_vector = generate_embedding(query_text)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        search_params=search_params,
        limit=top_n,
    )
    return results[0]

# Function to fetch text by ID from Milvus
def fetch_text_by_id(entity_id):
    try:
        query_results = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=f"id == {entity_id}",
            output_fields=["text"]
        )
        if query_results:
            return query_results[0].get("text", "Text not found")
    except Exception as e:
        print(f"Error querying Milvus for entity_id {entity_id}: {e}")
    return "Text not found"

# Function to process search results
def process_results(results):
    processed_results = []
    for result in results:
        result_data = {
            "id": result['id'],
            "distance": result['distance'],
            "text": fetch_text_by_id(result['id'])
        }
        processed_results.append(result_data)
    return processed_results

# Example usage
if __name__ == "__main__":
    while True:
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        # Search for similar text
        results = search_similar_text(query, top_n=3)
        processed_results = process_results(results)

        # Display results
        for idx, result in enumerate(processed_results, start=1):
            print(f"\nResult {idx}:")
            print(f"ID: {result['id']}, Distance: {result['distance']}, Text: {result['text']}")
