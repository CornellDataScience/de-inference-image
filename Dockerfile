FROM jfloff/alpine-python:3.6-slim

RUN apk add cmake

RUN pip install -r requirements.txt

ADD server.py .
CMD ["python", "server.py"]
