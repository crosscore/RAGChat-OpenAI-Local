FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

CMD ["streamlit", "run", "app/app.py"]
