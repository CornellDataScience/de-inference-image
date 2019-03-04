FROM python

RUN apt install cmake

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "server.py"]
