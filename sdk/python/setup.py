from setuptools import setup, find_packages

setup(
    name="el-sdk",
    version="1.0.0",
    description="Python SDK for EL (Eager Learner) - Question-Driven Knowledge Extraction Agent",
    author="EL Team",
    author_email="contact@example.com",
    url="https://github.com/your-org/el-knowledge",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
    ],
    python_requires=">=3.9",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
