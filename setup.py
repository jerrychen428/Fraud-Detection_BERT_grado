from setuptools import setup, find_packages

setup(
    name="fraud_detection_app",
    version="1.0.0",
    description="A BERT-based financial fraud detection app with Gradio interface.",
    author="jerrychen428",
    packages=find_packages(),
    install_requires=[
        "transformers==4.40.1",
        "torch==2.2.2",
        "scikit-learn==1.4.2",
        "pandas==2.2.2",
        "gradio==4.26.0",
        "fastapi==0.110.0",
        "uvicorn==0.22.0",
        "numpy==1.26.4",
        "accelerate==0.29.1"
    ],
    python_requires='=3.10.7',
)
