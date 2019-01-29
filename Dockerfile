FROM jfloff/alpine-python:3.6-slim

ADD server.py .
CMD ["python", "server.py"]
