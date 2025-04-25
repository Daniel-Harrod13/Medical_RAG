# Medical RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot specifically designed to answer questions based on a medical oncology handbook. It uses a local Large Language Model (LLM) and local embeddings, combined with a vector database, to provide contextually relevant answers through a web interface.

## Features

*   Question & Answering over PDF documents (`Data/` directory).
*   Utilizes local models for inference (Sentence Transformer for embeddings, Meditron 7B GGUF for generation).
*   FastAPI web interface for interaction.
*   Uses Qdrant as the vector database for efficient document retrieval.
*   Built with Langchain and CTransformers orchestration frameworks.

## Tech Stack

*   **Python 3.9+**
*   **Frameworks:** Langchain, FastAPI
*   **LLM:** Meditron 7B (via CTransformers)
*   **Embeddings:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (via Sentence Transformers)
*   **Vector Database:** Qdrant
*   **Web Server:** Uvicorn
*   **UI:** HTML, Bootstrap, JavaScript
*   **Core Libraries:** `torch`, `transformers`, `pypdf`, `qdrant-client`, `jinja2`

## Setup

### Prerequisites

1.  **Conda/Miniconda:** Ensure you have Conda installed to manage the environment.
2.  **Git:** To clone the repository.
3.  **Qdrant Instance:** You need a Qdrant vector database instance running. You can run it locally using Docker:
    ```bash
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ```
    Ensure Docker Desktop (or Docker Engine) is installed and running.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-name>
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n Medical_RAG python=3.9 -y
    conda activate Medical_RAG
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the LLM:**
    *   Download the `meditron-7b.Q4_K_M.gguf` model file. A common source is TheBloke's Hugging Face repository: [https://huggingface.co/TheBloke/meditron-7B-GGUF/tree/main](https://huggingface.co/TheBloke/meditron-7B-GGUF/tree/main)
    *   Place the downloaded `.gguf` file in the root directory of this project (`Medical_RAG/`).

5.  **Add Source Documents:**
    *   Place the PDF file(s) you want to query (e.g., `medical_oncology_handbook_june_2020_edition.pdf`) into the `Data/` directory.

## Running the Application

1.  **Run Data Ingestion:**
    *   This script processes the PDFs in the `Data/` directory, creates embeddings, and stores them in your Qdrant instance.
    ```bash
    python ingest.py
    ```
    *(Note: You only need to run this once, or again if you add/change documents in the `Data/` folder)*

2.  **Start the FastAPI Web Server:**
    ```bash
    uvicorn rag:app --reload
    ```
    *(Use `--reload` for development to automatically restart the server on code changes. Remove it for production.)*

3.  **Access the Chatbot:**
    *   Open your web browser and navigate to `http://127.0.0.1:8000`.

## How It Works

1.  **Ingestion (`ingest.py`):** Documents in `Data/` are loaded, split into manageable chunks, embedded using BiomedBERT, and stored in a Qdrant collection named `vector_database`.
2.  **Retrieval (`rag.py`):** When a query is submitted via the web UI:
    *   The query is embedded using the same BiomedBERT model.
    *   Qdrant is searched for the most similar document chunk(s) based on the query embedding.
3.  **Generation (`rag.py`):**
    *   The retrieved document chunk(s) (context) and the original query are formatted into a prompt.
    *   The prompt is sent to the local Meditron 7B LLM (via CTransformers).
    *   The LLM generates an answer based on the provided context and query.
4.  **Response:** The generated answer and the source document context are displayed in the web UI. 