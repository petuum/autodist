FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync \
    openssh-client \
    openssh-server

# The interactive-shell will break current AutoDist.
RUN sed -i '1s/^/[ -z "$PS1" ] \&\& return\n/' /etc/bash.bashrc

COPY dist /dist
RUN pip3 install dist/*.whl kazoo==2.6.1
