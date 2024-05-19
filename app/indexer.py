import pinecone
from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index = "ragchat-local"

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def index_qa_data(pinecone_index, qa_data):
    """
    質問と回答のテキストデータをPineconeにインデックスします。

    Args:
        pinecone_index (str): Pineconeのインデックス名
        qa_data (list): 質問と回答のデータリスト
            例: [
                {"question": "会社は何をしているのですか？", "answer": "当社は〇〇のサービスを提供しています。"},
                {"question": "サービスの特徴は？", "answer": "当社のサービスは〇〇です。"},
                ...
            ]
    """

    # Pineconeに接続
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pinecone.Index(pinecone_index)

    # データをインデックス
    for data in qa_data:
        question = data["question"]
        answer = data["answer"]
        # ベクトル化: OpenAI APIを使用
        question_embedding = get_embedding(question)
        # Pineconeにインデックス登録
        index.upsert(vectors=[(question_embedding, question)], metadata={"text": question, "answer": answer})

# ベクトル化関数
def get_embedding(text):
    # OpenAIのembeddings APIを使用
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text.replace("\n", " ")]
    )
    return response.data[0].embedding

# メイン処理
if __name__ == "__main__":
    # JSONファイルを読み込む
    with open("qa_data.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Pineconeにインデックスを作成
    index_qa_data(pinecone_index, qa_data)
