import os
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

from distutils.command.build_py import build_py as _build_py
from distutils.command.clean import clean as _clean


def generate_proto(source):
    """Invokes the Protocol Compiler to generate a _pb2.py from the given
    .proto file."""

    # Find the Protocol Compiler.
    if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
        protoc = os.environ['PROTOC']
    else:
        raise ValueError('Environment variable "PROTOC" is not a correct path for protoc executable.')

    output = source.replace(".proto", "_pb2.py")
    print("Generating %s..." % output)

    protoc_command = [protoc, "-I.", "--python_out=.", source]
    protoc_command = ' '.join(protoc_command)
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
                        filepath.endswith(".so") or filepath.endswith(".o"):
                    os.remove(filepath)
        # _clean is an old-style class, so super() doesn't work.
        _clean.run(self)


setup(
    name="autodist",
    version=os.environ.get('VERSION'),
    description="AutoDist",
    long_description="AutoDist is a scalable machine-learning engine to accelerate distributed training.",
    classifiers=[
        'Programming Language :: Python :: 3.6 :: Only',
    ],
    install_requires=[
        "numpy",
        "paramiko",
        "protobuf==3.11.0",
        "pyyaml",
        "tensorflow-gpu>=1.15,<2.1",
        "gast==0.2.2",
    ],
    extras_require={
        'test': [
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
    zip_safe=False,
)
