import os
import subprocess
from setuptools import setup, find_packages

from distutils.command.build_py import build_py as _build_py
from distutils.command.clean import clean as _clean


def generate_proto(source):
    """Invokes the Protocol Compiler to generate a _pb2.py from the given .proto file."""

    # Find the Protocol Compiler.
    if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
        protoc = os.environ['PROTOC']
    else:
        raise ValueError('Environment variable "PROTOC" is not a correct path for protoc executable.')

    output = source.replace(".proto", "_pb2.py")
    print("Generating %s..." % output)

    protoc_command = [protoc]
    if 'PROTOC_PLUGINS' in os.environ and os.path.exists(os.environ['PROTOC_PLUGINS']):
        protoc_command += ['--plugin='+os.environ['PROTOC_PLUGINS'],
                           '--doc_out=docs/usage',
                           '--doc_opt=docs/templates/proto.tmpl,proto_docgen.md']
    protoc_command += ["-I.", "--python_out=.", source]
    protoc_command = ' '.join(protoc_command)
    print(protoc_command)
    if subprocess.call(protoc_command, shell=True) != 0:
        exit(-1)


class build_py(_build_py):

    def run(self):
        generate_proto("autodist/proto/*.proto")

        # _build_py is an old-style class, so super() doesn't work.
        _build_py.run(self)


class clean(_clean):

    def run(self):
        # Delete generated files in the code tree.
        for (dirpath, dirnames, filenames) in os.walk("."):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if filepath.endswith("_pb2.py") or filepath.endswith(".pyc") or \
                        filepath.endswith(".so") or filepath.endswith(".o") or \
                        filepath.endswith("proto_docgen.md"):
                    os.remove(filepath)
        # _clean is an old-style class, so super() doesn't work.
        _clean.run(self)


setup(
    name="autodist",
    version=os.environ.get('VERSION'),
    author="Petuum Inc. & The AutoDist Authors",
    author_email="",
    description="AutoDist is a distributed deep-learning training engine.",
    long_description="""AutoDist provides a user-friendly interface to distribute 
                        the training of a wide variety of deep learning models
                        across many GPUs with scalability and minimal code change.""",
    url="https://github.com/petuum/autodist",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires=[
        "netifaces==0.10.9",
        "numpy",
        "paramiko",
        "protobuf==3.18.3",
        "pyyaml",
        # "tensorflow-gpu>=1.15,<=2.1",  # TF Version dynamic check at importing
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'pydocstyle',
            'prospector',
        ]
    },
    cmdclass={
        'build_py': build_py,
        'clean': clean,
    },
    packages=find_packages(),
    python_requires='>=3.6',
)
