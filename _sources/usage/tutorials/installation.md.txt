# Installation

#### Install From Release Wheel 

```bash
pip install autodist
```

#### Install From Latest Source

Before running AutoDist, we require a small compilation of our Protocol Buffers. 
To do so, you must first have [protoc installed](https://google.github.io/proto-lens/installing-protoc.html)
with the specific version indicated in `setup.py`.

You can run the following command :
```bash
git clone https://github.com/petuum/autodist.git
cd autodist

wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.0/protoc-3.11.0-linux-x86_64.zip
unzip protoc-3.11.0-linux-x86_64.zip
PROTOC=$(pwd)/bin/protoc python setup.py build  # compile our protobufs
pip install -e .[dev]  # install in development mode
```

To clean up any compiled files, run:
```bash
python setup.py clean --all
```
