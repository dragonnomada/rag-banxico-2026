FROM python:3.12.4-slim

WORKDIR /app

COPY app/* .

RUN python -m pip install -r requirements.txt

CMD ["python", "main.py"]