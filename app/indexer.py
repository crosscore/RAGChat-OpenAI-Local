from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import logging

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def index_qa_data(pinecone_index_name, qa_data):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        if pinecone_index_name not in pc.list_indexes().names():
            pc.create_index(
                name=pinecone_index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        index = pc.Index(pinecone_index_name)

        for i, data in enumerate(qa_data):
            question = data["question"]
            answer = data["answer"]
            question_embedding = get_embedding(question)
            answer_embedding = get_embedding(answer)
            index.upsert(vectors=[
                (f'question-{i}', question_embedding, {"text": question, "answer": answer}),
                (f'answer-{i}', answer_embedding, {"text": question, "answer": answer})
            ])
            logging.info(f"Indexed Q&A pair {i}")

    except Exception as e:
        logging.error(f"Error indexing data: {e}")

def get_embedding(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[text.replace("\n", " ")]
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return None

if __name__ == "__main__":
    try:
        with open("qa_data.json", "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        index_qa_data(PINECONE_INDEX_NAME, qa_data)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
