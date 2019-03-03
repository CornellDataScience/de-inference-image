FROM python:3.7-alpine

RUN apk add cmake

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "server.py"]
