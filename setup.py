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
    install_requires=[
        "numpy", 
        "paramiko", 
        "protobuf3",
        "pyyaml", 
        "tensorflow==2.0.0-beta1",
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'pydocstyle==3.0.0',
            'prospector==1.1.6.2',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
