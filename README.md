# SmartDesk AI

SmartDesk AI is a native Windows desktop assistant that lets office employees load TXT or PDF documents, ask questions locally through Ollama, and save chat conversations as UTF-8 text files.

## Features

- Native Tkinter desktop UI
- TXT and PDF document loading
- Local Ollama integration with the Llama 3.2 model
- Document Q&A with chunk-based context retrieval
- Document summarization
- Conversation saving to local text files
- Threaded background processing to keep the interface responsive

## Windows setup

1. Verify Python:
   ```powershell
   python --version
   ```
2. Verify Tkinter:
   ```powershell
   python -m tkinter
   ```
3. Install Ollama for Windows and start it.
4. Download the model:
   ```powershell
   ollama pull llama3.2
   ```
5. Verify the model is available:
   ```powershell
   ollama list
   ```
6. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
7. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
8. Run the application:
   ```powershell
   python app.py
   ```

## Project structure

- `app.py` – application entry point
- `src/gui.py` – Tkinter interface and event handling
- `src/document_reader.py` – TXT and PDF extraction
- `src/ollama_client.py` – Ollama HTTP client
- `src/document_qa.py` – chunking and retrieval logic
- `src/summarizer.py` – summarization prompts and orchestration
- `src/conversation_manager.py` – saved conversation formatting
- `src/config.py` – defaults and settings
- `tests/` – unit tests for core behavior

## Troubleshooting

- If Ollama is not reachable, confirm the service is running at `http://localhost:11434`.
- If the model is missing, run `ollama pull llama3.2`.
- If the app cannot read a PDF, it may be encrypted or contain scanned images without text.
- If the UI freezes, ensure the application is using worker threads for LLM calls.

## Privacy

This application processes document content locally through Ollama. It does not upload documents or conversations to cloud APIs.
