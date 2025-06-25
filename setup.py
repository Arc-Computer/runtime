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
    author="Arc Intelligence, Inc.",
    author_email="Jarrod@arc.computer",
    description="Lightweight Python interceptor that prevents AI agent failures in real-time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arc-computer/runtime",
    project_urls={
        "Bug Tracker": "https://github.com/arc-computer/runtime/issues",
        "Documentation": "https://github.com/arc-computer/runtime#readme",
        "Source Code": "https://github.com/arc-computer/runtime",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    package_data={
        "runtime": ["py.typed"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=[
        "wrapt>=1.14.0",
    ],
    extras_require={
        "telemetry": [
            "grpcio>=1.50.0",
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
            "opentelemetry-instrumentation>=0.41b0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "openai>=1.0.0",
            "grpcio>=1.50.0",
            "twine>=4.0.0",
            "wheel>=0.40.0",
            "build>=0.10.0",
        ],
    },
    keywords="ai, llm, openai, anthropic, reliability, monitoring, telemetry, interceptor",
)