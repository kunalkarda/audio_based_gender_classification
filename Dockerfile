FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 7860 8501
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 7860 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
