FROM bitnami/pytorch@sha256:148693c3a732e8b4da244fdb885bd9dfd66522e122a1ad73e4a1b62185e9c26f
LABEL authors="inaki-eab"

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r /app/requirements.txt

COPY . /app
