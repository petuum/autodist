# Installation

#### Install From Release Wheel 

```bash
pip install autodist
```

#### Install From Latest Source

Before running AutoDist, we require a small compilation of our Protocol Buffers. 
To do so, you must first have [protoc installed](https://google.github.io/proto-lens/installing-protoc.html)
with the specific version indicated in `setup.py`.

Then, you can run the following command :
```bash
git clone https://gitlab.int.petuum.com/internal/scalable-ml/autodist.git
cd autodist
PROTOC=`which protoc` python setup.py build  # compile our protobufs
pip install -e .  # install in development mode
```

To clean up any compiled files, run:
```bash
python setup.py clean --all

