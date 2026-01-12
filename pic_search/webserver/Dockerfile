FROM tensorflow/webserver-env:2.20.0

COPY ./src /app/src

RUN mkdir -p /tmp/search-images

CMD ["python", "app.py"]

# docker build   -t   milvusbootcamp/pic-search-webserver:2.20.0  .