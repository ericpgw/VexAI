from setuptools import setup, find_packages

setup(
    name="vexai",
    version="0.1.0",
    description="Brain-inspired AI enhancement system",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
    ],
)
