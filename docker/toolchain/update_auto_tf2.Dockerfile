FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync

RUN pip install pydocstyle prospector pytest pytest-cov