from setuptools import setup, find_packages
import os

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext

import sys

__version__ = "0.1"

from pathlib import Path
long_description = (Path(__file__).parent / "README.md").read_text()


def list_source_files(root_folder):
    cppfiles = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.cpp'):
                cppfiles.append(os.path.join(root, file))
    return cppfiles


ext_modules = [
    Pybind11Extension(
        "_bboptpy",
        list_source_files('py') + list_source_files('src') + list_source_files(''),
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name="bboptpy",
    version=__version__,
    author="Michael Gimelfarb",
    author_email="mgimelfarb@yahoo.ca",
    description="Powerful and scalable black-box optimization algorithms for Python and C++.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="LGPL-2.1 License",
    url="https://github.com/mike-gimelfarb/bboptpy",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.11.1', 'numpy'],
    python_requires=">=3.8",
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
