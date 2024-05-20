import pinecone
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
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def index_qa_data(pinecone_index_name, qa_data):
    """
    質問と回答のテキストデータをPineconeにインデックスします。

    Args:
        pinecone_index_name (str): Pineconeのインデックス名
        qa_data (list): 質問と回答のデータリスト
            例: [
                {"question": "会社は何をしているのですか？", "answer": "当社は〇〇のサービスを提供しています。"},
                {"question": "サービスの特徴は？", "answer": "当社のサービスは〇〇です。"},
                ...
            ]
    """

    try:
        # Pineconeに接続
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

        # Pineconeにインデックスが存在しない場合作成
        if pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(pinecone_index_name, dimension=1536)

        index = pinecone.Index(pinecone_index_name)

        # 質問と回答のデータセットをインデックス
        for i, data in enumerate(qa_data):
            question = data["question"]
            answer = data["answer"]
            # ベクトル化: OpenAI APIを使用
            question_embedding = get_embedding(question)
            answer_embedding = get_embedding(answer)
            # Pineconeにindex登録
            index.upsert(vectors=[
                (f'question-{i}', question_embedding, {"text": question, "answer": answer}),
                (f'answer-{i}', answer_embedding, {"text": question, "answer": answer})
            ])
            logging.info(f"Indexed Q&A pair {i}")

    except Exception as e:
        logging.error(f"Error indexing data: {e}")

# ベクトル化関数
def get_embedding(text):
    try:
        # OpenAIのembeddings APIを使用
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text.replace("\n", " ")]
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return None

# メイン処理
if __name__ == "__main__":
    # open json file
    try:
        with open("qa_data.json", "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        # Pineconeにインデックス作成
        index_qa_data(PINECONE_INDEX_NAME, qa_data)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
