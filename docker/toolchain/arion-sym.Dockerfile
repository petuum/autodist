# NOTE: this image should only be updated with new releases
FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync \
    openssh-client \
    openssh-server

# The interactive-shell will break current AutoDist.
RUN sed -i '1s/^/[ -z "$PS1" ] \&\& return\n/' /etc/bash.bashrc

COPY requirements* /
COPY setup.py /
RUN pip3 install --no-cache-dir -r requirements.txt

# TODO: move this to requirement later
RUN pip3 install kazoo==2.6.1

# TODO: decide the stable autodist version later
RUN pip3 install --index-url http://pypi.int.petuum.com:8080/simple --trusted-host pypi.int.petuum.com autodist==0.3.1.dev6
