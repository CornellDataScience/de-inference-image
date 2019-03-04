FROM python

RUN apt-get update && apt-get install -y cmake

RUN git clone https://github.com/davisking/dlib.git
RUN cd dlib && python3 setup.py install --no USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "server.py"]
