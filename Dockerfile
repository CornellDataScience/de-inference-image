FROM jfloff/alpine-python:3.6-slim

RUN apk add cmake

ADD server.py .
CMD ["python", "server.py"]
