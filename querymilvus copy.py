from openai import OpenAI
from pymilvus import MilvusClient

MODEL_NAME = "text-embedding-3-small"  # Same model used for embedding
DIMENSION = 1536  # Dimension of vector embedding
COLLECTION_NAME = "demo_collection"  # Milvus collection name

# Connect to OpenAI with API Key.
openai_client = OpenAI(api_key="x")

# Connect to Milvus
milvus_client = MilvusClient(uri="x", token="x")

# Query text
query_text = "Who conducted early research in artificial intelligence?"

# Generate embedding for the query text
query_vector = openai_client.embeddings.create(input=[query_text], model=MODEL_NAME).data[0].embedding

# Search in Milvus collection
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}  # Updated metric type to match collection
results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    search_params=search_params,  # Corrected argument name
    limit=3,  # Number of top results to retrieve
)

# Function to fetch text using ID by querying Milvus
def fetch_text_by_id(entity_id):
    print(f"Debug: Querying Milvus with expression: id == {entity_id}")  # Debug print
    try:
        # Correctly call the query method with proper arguments
        query_results = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=f"id == {entity_id}",  # Ensure this is passed correctly
            output_fields=["text"]  # Specify the field to retrieve
        )
        print(f"Debug: Query results: {query_results}")  # Debug print
        if query_results:
            return query_results[0].get("text", "Text not found")
    except Exception as e:
        print(f"Error querying Milvus for entity_id {entity_id}: {e}")
    return "Text not found"

# Process search results
processed_results = []
for result in results[0]:
    print(f"Debug: Processing result ID: {result['id']}")  # Debug print
    result_data = {
        "id": result['id'],
        "distance": result['distance'],
        "text": fetch_text_by_id(result['id'])  # Fetch text using the corrected function
    }
    processed_results.append(result_data)

# Example: Print processed results
for res in processed_results:
    print(f"ID: {res['id']}, Distance: {res['distance']}, Text: {res['text']}")

# Find the best result (smallest distance)
best_result = min(processed_results, key=lambda x: x['distance'])

# Display the best result
print("\nBest Result:")
print(f"ID: {best_result['id']}, Distance: {best_result['distance']}, Text: {best_result['text']}")

# Example: Use results for further processing
# For instance, you can save them to a file or use them in another function
# ...additional logic...