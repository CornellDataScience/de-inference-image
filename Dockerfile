FROM python

# RUN apk add cmake py-pip

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "server.py"]
