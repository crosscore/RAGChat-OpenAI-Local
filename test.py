import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key_from_env = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key_from_env
)

chat_completion_env = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "what is your name?",
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion_env.choices[0].message.content)
