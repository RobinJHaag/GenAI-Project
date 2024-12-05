import os
import requests
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document
from sentence_transformers import SentenceTransformer

# Hugging Face API Token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set.")
api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {hf_token}"}


# Step 1: Load documents from the 'documents' folder
def load_documents_from_file(file_path):
    """Reads the text file and returns a list of Document objects, each representing a process description."""
    documents = []
    with open(file_path, 'r') as file:
        content = file.read()
        process_descriptions = content.split(';')
        for idx, description in enumerate(process_descriptions):
            if description.strip():  # Ensure no empty descriptions are added
                documents.append(Document(content=description.strip(), id=str(idx)))
    return documents


# Step 2: Embed Documents
def embed_documents(documents, model):
    """Adds embeddings to the documents."""
    for doc in documents:
        doc.embedding = model.encode(doc.content)
    return documents


# Step 3: Initialize the Document Store
def initialize_document_store_with_embeddings(documents):
    """Initializes the document store with embeddings."""
    document_store = InMemoryDocumentStore(use_bm25=False)  # Disable BM25 as we'll use embeddings
    document_store.write_documents(documents)
    return document_store


# Step 4: Set up the Retriever
def initialize_retriever(document_store, model_name):
    """Initializes the embedding retriever with a model name."""
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=model_name,
        model_format="sentence_transformers"
    )
    return retriever


# Step 5: Query the Hugging Face model via the Inference API
def query_huggingface_model(prompt):
    """Queries the Hugging Face API with the provided prompt."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 2048,
            "top_p": 0.7,
        },
    }
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        completion = response.json()
        if isinstance(completion, list) and len(completion) > 0:
            generated_text = completion[0].get("generated_text", "")
            # Extract only the response by removing the input prompt from the output
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            return generated_text
        return ""
    else:
        return f"Error {response.status_code}: {response.text}"


# Step 6: Main Querying Function
def run_query_with_prompt_embedding(query, document_store, retriever, embedding_model):
    """Fetches relevant documents and queries the model."""
    # Retrieve documents for the plain text query
    retrieved_results = retriever.retrieve(query, top_k=2)

    # Log retrieved documents for debugging/optimization
    print("Retrieved Documents:")
    for result in retrieved_results:
        print(f"ID: {result.id}")
        print(f"Content: {result.content}\n")

    # Combine retrieved documents into a prompt
    retrieved_texts = [result.content for result in retrieved_results]
    combined_prompt = f"Here is some context: {' '.join(retrieved_texts)}\n\n{query}"

    # Query Hugging Face model
    model_output = query_huggingface_model(combined_prompt)

    return model_output, retrieved_results


if __name__ == "__main__":
    # Step 1: Load documents from the single file
    documents = load_documents_from_file("Documents/BPMNs.txt")

    # Step 2: Initialize the embedding model name
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Step 3: Embed documents using SentenceTransformer instance
    embedding_model = SentenceTransformer(embedding_model_name)
    documents = embed_documents(documents, embedding_model)

    # Step 4: Initialize the document store with embeddings
    document_store = initialize_document_store_with_embeddings(documents)

    # Step 5: Initialize retriever with model name
    retriever = initialize_retriever(document_store, embedding_model_name)

    # Step 6: Test the retriever with a query
    query = "Describe the airplane refueling process, use two sentences."

    # Step 7: Run the query with prompt embedding
    generated_response, retrieved_documents = run_query_with_prompt_embedding(query, document_store, retriever, embedding_model)

    # Step 8: Print retrieved documents
    print("\nRetrieved Documents:")
    for result in retrieved_documents:
        print(f"ID: {result.id}")
        print(f"Content: {result.content}\n")

    # Step 9: Print combined prompt (optional, for debugging purposes)
    print("\nCombined Prompt Sent to the LLM:")
    combined_prompt = f"Here is some context: {' '.join([doc.content for doc in retrieved_documents])}\n\n{query}"
    print(combined_prompt)

    # Step 10: Print the final response with clear separators
    print("\n" + "#" * 80)
    print("Generated Response:")
    print(generated_response)
    print("#" * 80 + "\n")

