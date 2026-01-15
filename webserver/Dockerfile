FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app/src

COPY ./requirements.txt /app/requirements.txt

# 安装 Python 依赖（PyTorch 已由基础镜像提供）
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# copy source code
COPY ./src /app/src

CMD ["python", "app.py"]
# docker build -t pytorch/webserver:2.9.1-cuda12.8-cudnn9-runtime -f ./Dockerfile .
