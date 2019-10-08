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
        "protobuf==3.9.1",
        "pyyaml",
        "tensorflow-gpu==2.0.0",
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
    packages=find_packages(),
    zip_safe=False,
)
