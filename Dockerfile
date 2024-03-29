# https://hub.docker.com/_/python
FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./


RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libaio1
RUN apt-get update && apt-get install -y gcc

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
