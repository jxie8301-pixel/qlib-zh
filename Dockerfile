FROM qlib-rdagent:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends vim docker.io && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple yahooquery akshare baostock

RUN ln -sf /usr/local/bin/python /usr/bin/python && \
    ln -sf /usr/local/bin/python3 /usr/bin/python3 && \
    ln -sf /usr/local/bin/qrun /usr/bin/qrun && \
    ln -sf /usr/local/bin/python /bin/python && \
    ln -sf /usr/local/bin/python3 /bin/python3 && \
    ln -sf /usr/local/bin/qrun /bin/qrun
