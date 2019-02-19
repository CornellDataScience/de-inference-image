FROM jfloff/alpine-python:3.6-slim

RUN sudo apt install -y cmake

ADD server.py .
CMD ["python", "server.py"]
