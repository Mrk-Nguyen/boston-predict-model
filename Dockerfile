FROM python:3.7-slim-buster

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update; apt-get install libgomp1

COPY . /app
WORKDIR /app

CMD ["python", "app.py"]