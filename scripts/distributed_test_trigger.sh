# setup the cuda version
export CUDA_HOME="/usr/local/cuda-10.0"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# activate the venv
source /home/autodist/venv/autodist/bin/activate
# run the test
pytest -s /home/autodist/autodist/tests/integration/test_dist.py