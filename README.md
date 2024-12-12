# Medical-Rag-Agent

This project is a Medical Information Assistant designed to help users with medical queries by providing relevant information and recommendations. The assistant uses various tools and a large language model to process and respond to user queries.

## Summary

The Medical Information Assistant is built using Python and Streamlit. It leverages several tools to provide accurate and helpful responses to medical queries. The assistant can summarize information, recommend actions, generate alternatives, make decisions, and retrieve information from the web.

## Files

### `agent.py`
This file contains the implementation of the `MedicalAgent` class, which is responsible for handling user queries and interacting with various tools to provide responses.

### `dataloader.py`
This file includes functions and classes for loading and processing data from various sources, such as JSON files.

### `langgraph.py`
This file contains the `visualize_chain` function, which uses Graphviz to visualize the chain of thought for the agent.

### `main.py`
This is the main entry point for running the Streamlit application. It imports and runs the `run_streamlit_app` function from `streamlit_app.py`.

### `streamlit_app.py`
This file contains the Streamlit application code. It sets up the user interface and handles user interactions.

### `tools.py`
This file defines various tools used by the `MedicalAgent`, such as `SummarizerTool`, `RecommenderTool`, `AlternativesGeneratorTool`, `LLMDecisionTool`, and `WebRetrievalTool`.

#### `SummarizerTool`
- **Name**: Summarizer
- **Description**: Summarizes drug information based on the JSON vector index.
- **Functionality**: This tool retrieves relevant documents from the FAISS index based on the user's query and provides a summarized version of the drug information.

#### `RecommenderTool`
- **Name**: Recommender
- **Description**: Recommends actions or treatments based on the user's symptoms or conditions.
- **Functionality**: This tool analyzes the user's input and suggests possible actions or treatments that could be beneficial.

#### `AlternativesGeneratorTool`
- **Name**: AlternativesGenerator
- **Description**: Generates alternative treatments or medications.
- **Functionality**: This tool provides alternative options for treatments or medications based on the user's current treatment plan or preferences.

#### `LLMDecisionTool`
- **Name**: LLMDecision
- **Description**: Makes decisions based on the large language model's analysis.
- **Functionality**: This tool uses the large language model to analyze the user's query and make informed decisions or recommendations.

#### `WebRetrievalTool`
- **Name**: WebRetrieval
- **Description**: Retrieves information from the web.
- **Functionality**: This tool searches the web for relevant information based on the user's query and provides the most relevant results.

## FAISS Data Index

The application uses FAISS (Facebook AI Similarity Search) to create and manage a data index for efficient similarity search. This allows the assistant to quickly retrieve relevant information from a large dataset based on user queries.
The index has been created using embeddings from HuggingfaceEmbeddings with the model "sentence-transformers/all-MiniLM-L6-v2".
They are in this repository.

## Large Language Model (LLM)

The application requires a large language model (LLM) running on either LM Studio or Ollama. In this case, the project is configured to use Ollama. Ensure that the LLM is running and accessible at the specified host and port.

## Product Catalogue

A product catalogue document that defines the scope of the index has been attached. This document provides detailed information about the data included in the FAISS index.


## Usage

To run the Streamlit application, execute the following command:

```sh
streamlit run main.py
```

Make sure to install the required dependencies listed in the `requirements.txt` file before running the application.

## Dependencies

Refer to the `requirements.txt` file for the complete list of dependencies.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.
```
