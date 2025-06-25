"""
Arc Runtime setup configuration
"""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="arc-runtime",
    version="0.1.0",
    author="Arc Team",
    author_email="team@arc.dev",
    description="Lightweight AI failure prevention system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arc-ai/arc-runtime",
    packages=find_packages(),
    package_data={
        "tests": ["fixtures/*.yaml"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "wrapt>=1.14.0",
    ],
    extras_require={
        "telemetry": ["grpcio>=1.50.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "openai>=1.0.0",
            "grpcio>=1.50.0",
        ],
    },
)