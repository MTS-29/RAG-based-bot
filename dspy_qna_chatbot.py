from dotenv import load_dotenv, find_dotenv
import dspy
from dspy.retrieve.clarifai_rm import ClarifaiRM
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Clarifai as clarifaivectorstore
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Clarifai
import os

load_dotenv(find_dotenv(), override=True)

from constants import APP_ID, MODEL_URL

PAT = "PAT_ID"
USER_ID = "USER_ID"


def loading_upserting_in_db(file):
    loader = Docx2txtLoader(file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    clarifai_vector_db = Clarifai(
        user_id=USER_ID,
        app_id=APP_ID,
        number_of_docs=len(docs),
        pat=PAT
    )
    return clarifai_vector_db


def loading_models():
    llm = dspy.Clarifai(model=MODEL_URL, api_key=PAT, n=2, inference_params={"max_tokens": 100, 'temperature': 0.6})
    retriever_model = ClarifaiRM(clarifai_user_id=USER_ID, clarfiai_app_id=APP_ID, clarifai_pat=PAT, k=2)

    dspy.settings.configure(lm=llm, rm=retriever_model)


class GenerateAnswer(dspy.Signature):
    """Think and Answer questions based on the context provided."""

    context = dspy.InputField(desc="may contain relevant facts about user query")
    question = dspy.InputField(desc="User query")
    answer = dspy.OutputField(desc="Answer in 100 to 150 Characters")


class RAG(dspy.Module):
    def __init__(self):
        super().__init__()

        self.retrieve = dspy.Retrieve()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)




if __name__ == "__main__":
    # File Name
    file = "66. GST Smart Guide.docx"
    clarifai_vector_db = loading_upserting_in_db(file)

    loading_models()

    # Ask any question you like to this RAG program.
    my_question = "What is gst ruling?"

    Rag_obj = RAG()
    predict = Rag_obj(my_question)

    # Print the contexts and the answer.
    print(f"Question: {my_question}")
    print(f"Predicted Answer: {predict}")
