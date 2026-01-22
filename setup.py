#!/usr/bin/env python3
"""
Setup script for VEGA-Verified.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="vega-verified",
    version="0.1.0",
    author="VEGA-Verified Team",
    description="Semantically Verified Neural Compiler Backend Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zachary-Lee-Jaeho/gensparktest",
    
    packages=find_packages(where="."),
    package_dir={"": "."},
    
    python_requires=">=3.8",
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "neural": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "vega-verify=src.cli:main",
            "vega-verified=src.main:main",
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
        "Topic :: Software Development :: Compilers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    keywords="compiler, backend, verification, neural, llvm, code-generation",
)
