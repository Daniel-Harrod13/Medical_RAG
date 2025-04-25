# Medical RAG Chatbot - Development Cookbook

This document provides a step-by-step guide detailing how this Medical RAG Chatbot application was built, including environment setup, script creation, and debugging steps encountered during development.

## Phase 1: Environment Setup & Dependencies

1.  **Requirement Definition (`requirements.txt`):**
    *   Identified necessary packages based on the project goal (Langchain, web framework, vector DB client, PDF reader, LLM loader, etc.).
    *   Created `requirements.txt` with the initial list:
        ```
        torch
        sentence_transformers
        transformers
        langchain
        fastapi
        uvicorn
        pypdf
        PyPDF2
        jinja2
        chroma # Initially included, but Qdrant was used in the final app
        qdrant-client
        ctransformers
        python-multipart
        aiofiles
        ```

2.  **Conda Environment Creation:**
    *   Created a dedicated Conda environment named `Medical_RAG` using Python 3.9:
        ```bash
        conda create -n Medical_RAG python=3.9 -y
        ```

3.  **Dependency Installation:**
    *   Activated the environment:
        ```bash
        conda activate Medical_RAG
        ```
    *   Installed packages from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Dependency Troubleshooting:**
    *   Encountered `ModuleNotFoundError` for `langchain_community` when running `ingest.py`. Installed it:
        ```bash
        pip install -U langchain-community
        ```
    *   Encountered `Import "qdrant_client" could not be resolved` in `retriever.py`. Found it wasn't installed correctly despite being in `requirements.txt`. Installed it explicitly:
        ```bash
        pip install qdrant-client
        ```

5.  **Qdrant Setup:**
    *   Ensured a Qdrant instance was running and accessible at `http://localhost:6333` (default port). Running via Docker is a common approach:
        ```bash
        docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
        ```
    *   *Initial development involved troubleshooting Docker Desktop installation and connection issues on macOS, including "malware" warnings (false positives due to signing issues) and daemon connection errors. This was resolved by reinstalling Docker and manually fixing helper tool binaries based on Docker documentation.* (This is context, Qdrant itself doesn't *require* Docker but it's a convenient way to run it).

## Phase 2: Data Ingestion (`ingest.py`)

1.  **Goal:** Load PDF documents from the `Data/` directory, split them into chunks, generate embeddings, and store them in the Qdrant vector database.

2.  **Key Imports:**
    *   `DirectoryLoader`, `PyPDFLoader` (from `langchain_community.document_loaders`): To load PDF files.
    *   `RecursiveCharacterTextSplitter` (from `langchain.text_splitter`): To split documents into smaller chunks.
    *   `SentenceTransformerEmbeddings` (from `langchain_community.embeddings`): To generate vector embeddings.
    *   `Qdrant` (from `langchain_community.vectorstores`): To interact with the Qdrant database.

3.  **Process:**
    *   Initialize `DirectoryLoader` to target the `Data/` directory, using `PyPDFLoader` for `.pdf` files.
    *   Load documents using `loader.load()`.
    *   Initialize `RecursiveCharacterTextSplitter` with appropriate `chunk_size` and `chunk_overlap`.
    *   Split loaded documents using `text_splitter.split_documents()`.
    *   Initialize `SentenceTransformerEmbeddings` with the chosen model: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`.
    *   Initialize the `Qdrant` vector store client, connecting to the running Qdrant instance (URL: `http://localhost:6333`) and specifying the collection name (`vector_database`).
    *   Use `Qdrant.from_documents()` to add the split text chunks and their generated embeddings to the specified collection.

## Phase 3: Retrieval Logic (`retriever.py`)

1.  **Goal:** Test the ability to connect to Qdrant and retrieve relevant document chunks based on a sample query. This script served as an intermediate step to verify the vector store population and retrieval mechanism before building the full web app.

2.  **Key Imports:**
    *   `Qdrant` (vector store class).
    *   `SentenceTransformerEmbeddings`.
    *   `QdrantClient` (low-level client).

3.  **Process:**
    *   Initialize the embedding model (`SentenceTransformerEmbeddings`) exactly as in `ingest.py`.
    *   Initialize the `QdrantClient`, specifying the Qdrant URL.
    *   Initialize the `Qdrant` vector store class, providing the client, collection name, and embedding model.
    *   Define a sample query string.
    *   Use `db.similarity_search_with_score()` to find the top `k` similar documents for the query.
    *   Iterate through the results and print the content and score.

4.  **Debugging:**
    *   Addressed `LangChainDeprecationWarning`s by updating imports from `langchain.*` to `langchain_community.*`.
    *   Fixed a `ValueError: Neither of embeddings or embedding_function is set` by changing the `Qdrant` initialization parameter from `embedding_function=embeddings` to `embeddings=embeddings` to match updated Langchain standards.

## Phase 4: RAG Application (`rag.py`)

1.  **Goal:** Create a FastAPI web application that integrates the retrieval (from Qdrant) and generation (using Meditron 7B) steps to provide answers based on user queries.

2.  **FastAPI Setup:**
    *   Import `FastAPI` and related modules (`Request`, `Form`, `HTMLResponse`, `Jinja2Templates`, `StaticFiles`, `jsonable_encoder`).
    *   Initialize the FastAPI app: `app = FastAPI()`.
    *   Configure Jinja2Templates to load HTML from the `templates` directory.
    *   Mount a `/static` directory for CSS/JS if needed (though CSS was included inline in `index.html`).

3.  **LLM Setup (CTransformers):**
    *   Define the path to the local LLM file: `local_llm = "meditron-7b.Q4_K_M.gguf"`.
    *   Define configuration parameters for `CTransformers` (`max_new_tokens`, `temperature`, `context_length`, etc.).
    *   Initialize `CTransformers`:
        ```python
        llm = CTransformers(
            model=local_llm,
            model_type="llama",
            # lib="avx2" was initially included but removed due to architecture mismatch
            **config
        )
        ```

4.  **Embeddings and Vector Store Setup:**
    *   Initialize the same `SentenceTransformerEmbeddings` model as used in ingestion.
    *   Initialize the `QdrantClient` and the `Qdrant` vector store class, connecting to the `vector_database` collection.

5.  **Langchain Orchestration:**
    *   Define a `PromptTemplate` suitable for RAG, including placeholders for `{context}` and `{question}`.
    *   Create a retriever from the Qdrant vector store using `db.as_retriever(search_kwargs={"k":1})` to fetch the single most relevant chunk.
    *   Initialize the `RetrievalQA` chain using `RetrievalQA.from_chain_type()`, providing:
        *   `llm`: The initialized CTransformers LLM.
        *   `chain_type="stuff"`: Standard chain type for putting context directly into the prompt.
        *   `retriever`: The Qdrant retriever.
        *   `return_source_documents=True`: To get the context document back.
        *   `chain_type_kwargs={"prompt": prompt}`: To use the custom prompt template.
        *   `verbose=True`: For debugging output during requests.

6.  **API Endpoints:**
    *   `@app.get("/")`: Serves the main `index.html` page using `templates.TemplateResponse`.
    *   `@app.post("/get_response")`: Handles user queries submitted via HTML form.
        *   Takes the `query` string from the form data.
        *   Calls the `qa` chain with the user's query: `response = qa(query)`.
        *   Extracts the answer (`result`) and source document details from the chain's response.
        *   Formats the extracted data into a JSON structure.
        *   Uses `jsonable_encoder(json.dumps(...))` to handle potential double encoding needed when sending complex data back via FastAPI `Response`.
        *   Returns the JSON data in a FastAPI `Response` object.

7.  **Debugging:**
    *   Resolved `Attribute "app" not found in module "rag"` error caused by the `rag.py` file being accidentally emptied. Restored content and saved.
    *   Resolved `huggingface_hub.errors.RepositoryNotFoundError` by changing `local_llm` from a non-existent repo ID (`Meditron-7b-v3.Q4_K_M.gguf`) to the correct local filename (`meditron-7b.Q4_K_M.gguf`) after downloading the model.
    *   Resolved `OSError: ... incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64')` by removing the explicit `lib="avx2"` parameter from the `CTransformers` initialization, allowing it to auto-detect the correct library for Apple Silicon (ARM64).

## Phase 5: Frontend (`templates/index.html`)

1.  **Goal:** Create a simple web page for users to input queries and see responses.

2.  **Structure:**
    *   Basic HTML5 structure.
    *   Linked Bootstrap CSS for styling.
    *   Included inline CSS within `<style>` tags for custom appearance.
    *   An accordion section to display app information.
    *   A `textarea` for user input.
    *   A `button` to submit the query.
    *   A `div` to display the response.

3.  **JavaScript Logic:**
    *   Added an event listener to the submit button.
    *   On click, it retrieves the user input text.
    *   Displays a loading indicator (Bootstrap spinner).
    *   Uses the `fetch` API to send a POST request to the `/get_response` endpoint on the FastAPI server, sending the query in `FormData`.
    *   Handles the response:
        *   Checks if the network response is okay.
        *   Parses the JSON response (including handling the double encoding from the backend).
        *   Displays the answer, context (`source_document`), and source file (`doc`) in the response `div`.
        *   Includes basic error handling to display messages if the fetch fails or the server returns an error.

4.  **Styling:**
    *   Initial basic styling was applied.
    *   Later updated with a more modern look, using a light background and royal blue accents for a medical theme, improving padding, borders, and shadows for better visual structure.
    *   Troubleshooting involved fixing a blank page issue caused by an empty `index.html` file. Restored content and saved.

## Phase 6: Running the Application

1.  **Ingestion:**
    ```bash
    conda activate Medical_RAG
    python ingest.py
    ```

2.  **Web Server:**
    ```bash
    conda activate Medical_RAG
    uvicorn rag:app --reload
    ```

3.  **Access:** Open `http://127.0.0.1:8000` in a web browser. 