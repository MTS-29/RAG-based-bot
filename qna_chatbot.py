from dotenv import load_dotenv, find_dotenv
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import warnings
from constants import INDEX_NAME

warnings.filterwarnings("ignore")

load_dotenv(find_dotenv(), override=True)

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
embedding = OpenAIEmbeddings(model="text-embedding-3-small")


def pinecone_index_creation(embedding_model):
    if INDEX_NAME in pc.list_indexes().names():
        index_flag = True
        print("Index already exists")
        index_vector_store = Pinecone(index_name=INDEX_NAME, embedding=embedding_model)
    else:
        index_flag = False
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
        index_vector_store = Pinecone(index_name=INDEX_NAME, embedding=embedding_model)
    return index_vector_store, index_flag


def load_document(doc_file):
    loader = Docx2txtLoader(doc_file)
    data = loader.load()
    return data


def chuck_data(data, chunk_size=600, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    data_chunks = text_splitter.split_documents(data)
    return data_chunks


def upsert_chunks_to_pinecone(data_chunks, embedding_model, index_name_pinecone, index_flag):
    if not index_flag:
        print(f"Going to add {len(data_chunks)} to Pinecone")
        index_vector_store = PineconeVectorStore.from_documents(
            data_chunks, embedding_model, index_name=index_name_pinecone
        )
        print("****Loading to vectorstore done ***")
        return index_vector_store
    else:
        index_vector_store = PineconeVectorStore.from_existing_index(index_name=index_name_pinecone,
                                                                     embedding=embedding_model)
        return index_vector_store


def query_llm(search_query, index_vector_store):
    llm = ChatOpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = index_vector_store.similarity_search(query=search_query)
    return chain.invoke({"input_documents": docs, 'question': search_query}), 200


if __name__ == "__main__":
    vector_store, flag = pinecone_index_creation(embedding)

    file = "66. GST Smart Guide.docx"
    document_data = load_document(file)
    chunks = chuck_data(document_data)

    vector_store_index = upsert_chunks_to_pinecone(chunks, embedding, INDEX_NAME, flag)
    query = input("Enter your query: ")
    answer = query_llm(query, vector_store_index)
    print(answer[0]['output_text'])
