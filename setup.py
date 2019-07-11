import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

setup(
    name="autodist",
    version=os.environ.get('VERSION'),
    description="Scalable ML Engine",
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
    ],
    install_requires=["paramiko", "numpy", "pyyaml", "protobuf"],
    packages=find_packages(),
    zip_safe=False,
)
