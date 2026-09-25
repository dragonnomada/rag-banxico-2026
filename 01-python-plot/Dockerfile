FROM python:3.12.4-slim

WORKDIR /app

COPY app/* .

RUN python -m pip install numpy pandas matplotlib seaborn

CMD ["python", "main.py"]