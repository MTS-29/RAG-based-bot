from dotenv import load_dotenv, find_dotenv
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from constants import INDEX_NAME

# Loading the env variables
load_dotenv(find_dotenv(), override=True)

# Initialization of Pinecone database
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

# Initialization of embedding model
embedding = OpenAIEmbeddings(model="text-embedding-3-small")


# Pinecone Index Creation
def pinecone_index_creation(embedding_model):
    if INDEX_NAME in pc.list_indexes().names():
        index_flag = True
        print("************** Index already exists **************")
        index_vector_store = Pinecone(index_name=INDEX_NAME, embedding=embedding_model)
    else:
        index_flag = False
        print("************** Creating new Pinecone Index **************")
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


# Loading the document
def load_document(doc_file):
    loader = Docx2txtLoader(doc_file)
    data = loader.load()
    return data


# Chunking the loaded document
def chuck_data(data, chunk_size=600, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    data_chunks = text_splitter.split_documents(data)
    return data_chunks


# inserting embeddings into the Pinecone Index
def upsert_chunks_to_pinecone(data_chunks, embedding_model, index_name_pinecone, index_flag):
    # Check if document is already indexed
    if not index_flag:
        print(f"************** Going to add {len(data_chunks)} chunks to Pinecone **************")
        index_vector_store = PineconeVectorStore.from_documents(
            data_chunks, embedding_model, index_name=index_name_pinecone
        )
        print("************** Loading to vectorstore done **************")
        return index_vector_store
    else:
        index_vector_store = PineconeVectorStore.from_existing_index(index_name=index_name_pinecone,
                                                                     embedding=embedding_model)
        return index_vector_store


# Querying with a search text from the vectors inside pinecone index
def query_llm(search_query, index_vector_store):

    llm = ChatOpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])

    docs = index_vector_store.similarity_search(query=search_query)

    chat_template = ChatPromptTemplate.from_messages([
        ("system", """After reading the document, give a clear and fact-based answer to the question. 
        Each question stands alone. Your answer should use info from the document.
        Don't add anything that's not in the text or make guesses about what's not there. 
        {docs}"""),
        ("human", "{user_input}")
    ])

    prompt = chat_template.invoke({
        "user_input": search_query,
        "docs": docs
    })
    response = llm.invoke(prompt)
    return response


if __name__ == "__main__":
    vector_store, flag = pinecone_index_creation(embedding)

    # File Name
    file = "66. GST Smart Guide.docx"

    document_data = load_document(file)
    chunks = chuck_data(document_data)

    vector_store_index = upsert_chunks_to_pinecone(chunks, embedding, INDEX_NAME, flag)

    # Input for the search text
    query = input("Enter your query: ")
    answer = query_llm(query, vector_store_index)

    print(answer.content)
