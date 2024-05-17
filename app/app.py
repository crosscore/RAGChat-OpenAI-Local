import streamlit as st
import os
from datetime import timedelta
from dotenv import load_dotenv
from pinecone import pinecone, ServerlessSpec
from openai import OpenAI
from momento import CacheClient, Configurations, CredentialProvider
from momento.responses import CacheGet, CacheSet

load_dotenv()

# Initialize Pinecone
pc = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
pinecone_index = "ragchat-local"

# Initialize OpenAI
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Momento
momento_auth_token = CredentialProvider.from_environment_variable('MOMENTO_API_KEY')
ttl = timedelta(seconds=int(os.getenv('MOMENTO_TTL_SECONDS', '600')))
cache_client = CacheClient.create(
    configuration=Configurations.Laptop.v1(),
    credential_provider=momento_auth_token,
    default_ttl=ttl
)
cache_name = "chatbot-cache"

def get_cached_answer(question):
    try:
        cached_response = cache_client.get(cache_name, question)
        if isinstance(cached_response, CacheGet.Hit):
            return cached_response.value_string
    except Exception as e:
        print(f"Cache get error: {e}")
    return None

def cache_answer(question, answer):
    try:
        cache_client.set(cache_name, question, answer)
    except Exception as e:
        print(f"Cache set error: {e}")

def get_embeddings(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text.replace("\n", " ")]
    )
    return response.data[0].embedding

st.title("Question Answering Chatbot")
user_input = st.text_input("Please enter your question:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Submit"):
    if user_input:
        cached_answer = get_cached_answer(user_input)
        if cached_answer:
            st.session_state.chat_history.append((user_input, cached_answer + " (cached)"))
        else:
            user_input_vector = get_embeddings(user_input)

            # Check if the index exists
            if pinecone_index in pc.list_indexes():
                index = pc.Index(pinecone_index, host=os.getenv("PINECONE_HOST"))
                query_result = index.query(vector=user_input_vector, top_k=1)

                if query_result['matches']:
                    closest_question = query_result['matches'][0]['metadata']['text']
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": closest_question}
                        ]
                    )
                    answer = response.choices[0].message.content
                    st.session_state.chat_history.append((user_input, answer))
                    cache_answer(user_input, answer)
                else:
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": user_input}
                        ]
                    )
                    answer = response.choices[0].message.content
                    st.session_state.chat_history.append((user_input, answer))
                    cache_answer(user_input, answer)
            else:
                # If the index does not exist, fallback to OpenAI only
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input}
                    ]
                )
                answer = response.choices[0].message.content
                st.session_state.chat_history.append((user_input, answer))
                cache_answer(user_input, answer)
    else:
        st.write("Please enter your question")

if st.session_state.chat_history:
    for q, a in st.session_state.chat_history:
        st.write(f"Q: {q}")
        st.write(f"A: {a}")
