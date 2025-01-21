from setuptools import setup, find_packages

setup(
    name="legal_doc_extractor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "PyPDF2==3.0.1",
        "transformers==4.36.2",
        "spacy==3.7.2",
        "fpdf==1.7.2",
        "streamlit==1.29.0",
        "torch==2.1.2",
        "protobuf==4.25.1"
    ],
)