FROM jfloff/alpine-python:3.6-slim

RUN apk add cmake

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "server.py"]
