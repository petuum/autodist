# This is a Docker Image recommended for AutoDist users.
# For more usage instructions, please refer to docs/usage/tutorials/docker.md

ARG  TF_VERSION=2.0.1
FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN rm -rf /etc/bash.bashrc

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        git \
        curl \
        vim \
        wget \
        unzip

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

## Install Dependencies
#RUN pip install future typing wheel pydocstyle prospector pytest pytest-cov kazoo==2.6.1
#RUN echo 'import coverage; coverage.process_startup()' >> /usr/lib/python3.6/sitecustomize.py
#
## Install OpenSSH to communicate between containers
#RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
#    mkdir -p /var/run/sshd
#
## Setup SSH Daemon
#RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
#RUN sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
#
## Allow OpenSSH to talk to containers without asking for confirmation
#RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
#    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
#    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN apt-get install -y golang-go && go get -u github.com/pseudomuto/protoc-gen-doc/cmd/protoc-gen-doc

ENV PROTOC_PLUGINS=/root/go/bin/protoc-gen-doc


RUN pip install sphinx recommonmark sphinx-git sphinx-rtd-theme dhubbard-sphinx-markdown-tables

RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.0/protoc-3.11.0-linux-x86_64.zip

WORKDIR /autodist

ENTRYPOINT mv /protoc-3.11.0-linux-x86_64.zip . && unzip protoc-3.11.0-linux-x86_64.zip && \
    PROTOC=/autodist/bin/protoc HOME=/autodist python setup.py build && \
    rm protoc-3.11.0-linux-x86_64.zip && rm -fr /autodist/bin /autodist/include /autodist/readme.txt && \
    pip install -e . && \
    cd docs && cp ../README.md README.md && sed -i.bak 's+docs/++g' README.md && \
    make clean && make apidoc && make html
