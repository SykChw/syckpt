"""Setup for syckpt package - Git-like experiment tracking for DL."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text()
    if (this_directory / "README.md").exists()
    else ""
)

setup(
    name="syckpt",
    version="0.0.3",
    description="Git-like experiment tracking for deep learning with LSH hashing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sayak Chowdhury",
    author_email="sayak.iiitb@gmail.com",
    url="https://github.com/sykchw/syckpt",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "safetensors>=0.4.0",
        "fsspec>=2023.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "twine>=4.0.0",
            "build>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning deep-learning experiment-tracking checkpoint reproducibility lsh",
    project_urls={
        "Bug Reports": "https://github.com/sykchw/syckpt/issues",
        "Source": "https://github.com/sykchw/syckpt",
    },
)
