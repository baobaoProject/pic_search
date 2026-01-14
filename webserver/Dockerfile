FROM tensorflow/tensorflow:2.20.0

WORKDIR /app/src
ENV TF_XLA_FLAGS=--tf_xla_cpu_global_jit

COPY ./requirements.txt /app/requirements.txt
COPY ./models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 /app/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
RUN mkdir -p /root/.keras/models && mv /app/models/* /root/.keras/models/

# 降级protobuf以解决兼容性问题
RUN pip install protobuf==3.20.*

RUN pip install --upgrade pip && pip install --no-cache-dir  -r  /app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com && \
    pip cache purge && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# copy source code
COPY ./src /app/src

CMD ["python", "app.py"]
# docker build   -t   tensorflow/webserver:2.20.0 -f ./Dockerfile  .
