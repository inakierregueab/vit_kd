FROM nvcr.io/nvidia/pytorch:23.03-py3

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r /app/requirements.txt

COPY . /app
