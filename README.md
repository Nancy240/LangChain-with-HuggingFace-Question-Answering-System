# LangChain with HuggingFace Question-Answering System

This repository contains a Jupyter Notebook (`LangChain_HuggingFace_QA.ipynb`) that demonstrates the integration of **LangChain** with **HuggingFace** models for building a question-answering system. The notebook uses two approaches: the `LlamaCpp` class with the Mistral-7B model for local inference and the `HuggingFacePipeline` with the GPT-2 model for text generation. It showcases how to set up a language model pipeline, create prompt templates, and answer questions like "What is Machine Learning?" and "Who won the Cricket World Cup in 2011?".

## Project Overview
The notebook implements a question-answering system with the following features:
- **Model Integration**: Uses `TheBloke/Mistral-7B-v0.1-GGUF` (quantized with Q4_K_M) via `LlamaCpp` and `gpt2` via `HuggingFacePipeline`.
- **Prompt Templates**: Utilizes LangChain's `PromptTemplate` and `LLMChain` (with a note on its deprecation) for structured question-answering.
- **GPU Support**: Leverages GPU acceleration with `device_map="auto"` for efficient inference.
- **Example Queries**: Demonstrates responses to questions about machine learning and the 2011 Cricket World Cup.

The system answers queries by combining LangChain's chaining capabilities with HuggingFace's model ecosystem, making it a versatile example for NLP tasks.

## Prerequisites
To run the notebook, you need:
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- GPU (optional, for faster inference; NVIDIA Tesla T4 used in the original setup)
- A HuggingFace API token for accessing models (stored securely in Kaggle secrets in the original notebook)

### Required Libraries
Install the required dependencies using the following command:
```bash
pip install langchain langchain-huggingface transformers accelerate bitsandbytes
