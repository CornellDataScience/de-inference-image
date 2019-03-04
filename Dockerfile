FROM python

RUN apt-get update && apt-get install -y cmake

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "server.py"]
