FROM python:3.12.4-slim

WORKDIR /app

COPY app/* .

RUN python -m pip install -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0"]