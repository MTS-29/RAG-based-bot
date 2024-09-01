from dotenv import load_dotenv, find_dotenv
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from constants import INDEX_NAME

load_dotenv(find_dotenv(), override=True)

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
embedding = OpenAIEmbeddings(model="text-embedding-3-small")


def pinecone_index_creation(embedding_model):
    if INDEX_NAME in pc.list_indexes().names():
        print("Index already exists")
        vector_store_pinecone = Pinecone(index_name=INDEX_NAME, embedding=embedding_model)
    else:
        print("Creating new Pinecone Index")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        vector_store_pinecone = Pinecone(index_name=INDEX_NAME, embedding=embedding_model)
    return vector_store_pinecone


def load_document(doc_file):
    loader = Docx2txtLoader(doc_file)
    data = loader.load()
    return data


def chuck_data(data, chunk_size=600, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    data_chunks = text_splitter.split_documents(data)
    return data_chunks


def upsert_chunks_to_pinecone(data_chunks, embedding_model, index_name_pinecone):
    print(f"Going to add {len(data_chunks)} to Pinecone")
    # PineconeVectorStore.from_documents(
    #     data_chunks, embedding_model, index_name=index_name_pinecone
    # )
    print("****Loading to vectorstore done ***")


def query_pinecone(query_text, embedding_model, index_name_pinecone, top_k=5):
    """
    Queries the Pinecone index with a given text.

    Parameters:
    - query_text (str): The text to query.
    - embedding_model (OpenAIEmbeddings): The embedding model used for querying.
    - index_name_pinecone (str): The Pinecone index name.
    - top_k (int): The number of top results to return.

    Returns:
    - List of results with the most relevant documents.
    """
    # Create a Pinecone vector store instance for querying
    vector_store_pinecone = PineconeVectorStore(index_name=index_name_pinecone, embedding=embedding_model)

    # Debugging: Print type of query_text
    print(f"Type of query_text: {type(query_text)}")

    # Convert the query text into an embedding
    try:
        query_embedding = embedding_model.embed_query(query_text)
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

    # Debugging: Print type of query_embedding
    print(f"Type of query_embedding: {type(query_embedding)}")

    # Query the Pinecone index
    try:
        results = vector_store_pinecone.similarity_search(query_embedding, k=top_k)
    except Exception as e:
        print(f"Error during Pinecone similarity search: {e}")
        return []

    # Return the results
    return results


if __name__ == "__main__":
    vector_store = pinecone_index_creation(embedding)

    file = "66. GST Smart Guide.docx"
    document_data = load_document(file)
    chunks = chuck_data(document_data)

    upsert_chunks_to_pinecone(chunks, embedding, INDEX_NAME)

    query = "What is GST?"
    results = query_pinecone(query, embedding, INDEX_NAME)

    # Print the query results
    print("Query Results:")
    for result in results:
        print(result)