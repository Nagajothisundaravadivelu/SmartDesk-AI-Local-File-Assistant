from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .config import DEFAULT_MODEL, OLLAMA_BASE_URL, PROJECT_NAME, SUBTITLE
from .conversation_manager import ConversationManager
from .document_qa import build_question_prompt
from .document_reader import DocumentError, load_document
from .ollama_client import OllamaClient, OllamaError
from .summarizer import summarize_document


class SmartDeskAIApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"{PROJECT_NAME} - Local File Assistant")
        self.root.geometry("1180x760")
        self.root.minsize(950, 620)
        self.root.configure(bg="#edf3f8")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.logger = logging.getLogger("smartdesk_ai")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        self.ollama_client = OllamaClient(base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL)
        self.conversation_manager = ConversationManager()
        self.current_document = None
        self.is_busy = False
        self.worker_queue: queue.Queue[dict] = queue.Queue()

        self._build_ui()
        self._append_message(
            "assistant",
            "Welcome to SmartDesk AI. Load a PDF or TXT document and ask a question when you are ready.",
        )
        self._queue_status_check()
        self.root.after(200, self._poll_queue)

    def _build_ui(self) -> None:
        self.sidebar = ttk.Frame(self.root, padding=(18, 18))
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.main_panel = ttk.Frame(self.root, padding=(18, 18))
        self.main_panel.grid(row=0, column=1, sticky="nsew")

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.sidebar.columnconfigure(0, weight=1)
        self.main_panel.columnconfigure(0, weight=1)
        self.main_panel.rowconfigure(1, weight=1)

        header = ttk.Frame(self.sidebar)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(header, text=PROJECT_NAME, font=("Segoe UI", 20, "bold")).grid(sticky="w")
        ttk.Label(header, text=SUBTITLE, foreground="#4a5d6a", font=("Segoe UI", 10)).grid(sticky="w", pady=(2, 0))

        self.status_var = tk.StringVar(value="Checking Ollama...")
        self.model_var = tk.StringVar(value=f"Model: {DEFAULT_MODEL}")
        self.document_name_var = tk.StringVar(value="No document loaded")
        self.document_path_var = tk.StringVar(value="Selected path: -")
        self.document_meta_var = tk.StringVar(value="Document details: Not available")

        self.browse_btn = ttk.Button(self.sidebar, text="Browse Document", command=self.browse_document)
        self.browse_btn.grid(row=1, column=0, sticky="ew", pady=(4, 10))

        ttk.Label(self.sidebar, textvariable=self.document_name_var, font=("Segoe UI", 11, "bold")).grid(sticky="w", pady=(0, 4))
        ttk.Label(self.sidebar, textvariable=self.document_path_var, wraplength=260, justify="left").grid(sticky="w", pady=(0, 4))
        ttk.Label(self.sidebar, textvariable=self.document_meta_var, wraplength=260, justify="left").grid(sticky="w", pady=(0, 8))

        self.clear_btn = ttk.Button(self.sidebar, text="Clear Document", command=self.clear_document)
        self.clear_btn.grid(row=2, column=0, sticky="ew", pady=(4, 8))

        self.new_chat_btn = ttk.Button(self.sidebar, text="New Chat", command=self.new_chat)
        self.new_chat_btn.grid(row=3, column=0, sticky="ew", pady=(8, 8))

        self.save_btn = ttk.Button(self.sidebar, text="Save Conversation", command=self.save_conversation)
        self.save_btn.grid(row=4, column=0, sticky="ew", pady=(0, 12))

        info_frame = ttk.LabelFrame(self.sidebar, text="Local AI Status", padding=(10, 8))
        info_frame.grid(row=5, column=0, sticky="ew")
        ttk.Label(info_frame, textvariable=self.status_var, foreground="#2a7f62").grid(sticky="w")
        ttk.Label(info_frame, textvariable=self.model_var, foreground="#44576a").grid(sticky="w", pady=(4, 0))

        self.chat_panel = ttk.Frame(self.main_panel)
        self.chat_panel.grid(row=0, column=0, sticky="nsew")
        self.chat_panel.columnconfigure(0, weight=1)
        self.chat_panel.rowconfigure(0, weight=1)

        self.chat_box = scrolledtext.ScrolledText(
            self.chat_panel,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 10),
            padx=12,
            pady=10,
            bg="#ffffff",
            fg="#1e2d39",
            height=22,
        )
        self.chat_box.grid(row=0, column=0, sticky="nsew")
        self.chat_box.tag_configure("user", foreground="#123a52", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("assistant", foreground="#2d6a4f", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("system", foreground="#7a6c57", font=("Segoe UI", 10, "italic"))

        self.input_frame = ttk.Frame(self.main_panel)
        self.input_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.input_frame.columnconfigure(0, weight=1)

        self.input_box = tk.Text(self.input_frame, height=4, wrap="word", font=("Segoe UI", 10), padx=10, pady=8)
        self.input_box.grid(row=0, column=0, sticky="ew")
        self.input_box.insert("1.0", "Ask a question about your document...")
        self.input_box.configure(fg="#7a8187")
        self.input_box.bind("<FocusIn>", self._clear_placeholder)
        self.input_box.bind("<KeyRelease>", self._handle_keypress)

        button_bar = ttk.Frame(self.input_frame)
        button_bar.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.send_btn = ttk.Button(button_bar, text="Send", command=self.ask_question)
        self.send_btn.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.summarize_btn = ttk.Button(button_bar, text="Summarize Document", command=self.summarize_document)
        self.summarize_btn.grid(row=1, column=0, sticky="ew")

        self._set_busy_state(False)

    def on_close(self) -> None:
        self.root.destroy()

    def _clear_placeholder(self, event) -> None:
        value = self.input_box.get("1.0", "end").strip()
        if value == "Ask a question about your document...":
            self.input_box.delete("1.0", "end")
            self.input_box.configure(fg="#1e2d39")

    def _handle_keypress(self, event) -> None:
        if event.keysym == "Return" and not event.state & 0x1:
            self.ask_question()
            return "break"
        if event.keysym == "Return" and event.state & 0x1:
            return None

    def _append_message(self, role: str, message: str) -> None:
        normalized = message.strip()
        if not normalized:
            return
        if role in {"user", "assistant"}:
            self.conversation_manager.add_message(role, normalized)
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"\n{normalized}\n", role)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _set_busy_state(self, busy: bool) -> None:
        self.is_busy = busy
        buttons = [self.send_btn, self.summarize_btn, self.browse_btn, self.clear_btn, self.save_btn, self.new_chat_btn]
        state = "disabled" if busy else "normal"
        for button in buttons:
            button.configure(state=state)

    def _queue_status_check(self) -> None:
        self._start_background_task("status_check", {})

    def _poll_queue(self) -> None:
        while True:
            try:
                item = self.worker_queue.get_nowait()
            except queue.Empty:
                break

            item_type = item.get("type")
            if item_type == "ollama_status":
                connected = bool(item.get("connected"))
                available = bool(item.get("model_available"))
                if connected and available:
                    self.status_var.set("Ollama ready")
                    self.model_var.set(f"Model: {DEFAULT_MODEL}")
                elif connected:
                    self.status_var.set("Ollama reachable but model missing")
                    self.model_var.set(f"Model: {DEFAULT_MODEL}")
                else:
                    self.status_var.set("Ollama unavailable")
                    self.model_var.set(f"Model: {DEFAULT_MODEL}")
                continue

            if item_type == "document_loaded":
                document = item.get("document")
                self.current_document = document
                self.document_name_var.set(document.file_name)
                self.document_path_var.set(f"Selected path: {document.file_path}")
                self.document_meta_var.set(f"File type: {document.extension.upper()} | Characters: {document.char_count}")
                self._append_message("assistant", f"Document loaded successfully: {document.file_name}. You can now ask questions or summarize it.")
                continue

            if item_type == "assistant_response":
                question = item.get("question")
                if question:
                    self._append_message("user", f"You: {question}")
                self._append_message("assistant", item.get("content", ""))
                continue

            if item_type == "summary_response":
                self._append_message("assistant", item.get("content", ""))
                continue

            if item_type == "error":
                self._append_message("system", f"Error: {item.get('message', 'Something went wrong.')}")
                continue

            if item_type == "task_finished":
                self._set_busy_state(False)
                continue

        self.root.after(200, self._poll_queue)

    def _start_background_task(self, task_name: str, payload: dict) -> None:
        if self.is_busy and task_name not in {"status_check"}:
            return

        self._set_busy_state(True)

        def worker() -> None:
            try:
                if task_name == "status_check":
                    connected = self.ollama_client.check_connection()
                    available = connected and self.ollama_client.check_model_available()
                    self.worker_queue.put({"type": "ollama_status", "connected": connected, "model_available": available})
                elif task_name == "load_document":
                    document = load_document(payload["path"])
                    self.worker_queue.put({"type": "document_loaded", "document": document})
                elif task_name == "ask_question":
                    question = payload["question"].strip()
                    if not self.current_document:
                        raise ValueError("Please select a document before asking a question.")
                    if not question:
                        raise ValueError("Please enter a question before sending.")
                    prompt = build_question_prompt(question, self.current_document.text)
                    response = self.ollama_client.generate_chat_response(
                        prompt,
                        system_prompt="You are a careful local document assistant. Base your answer only on the supplied document content.",
                    )
                    self.worker_queue.put({"type": "assistant_response", "question": question, "content": response})
                elif task_name == "summarize":
                    if not self.current_document:
                        raise ValueError("Please select a document before summarizing.")
                    summary = summarize_document(self.current_document.text, self.ollama_client)
                    self.worker_queue.put({"type": "summary_response", "content": summary})
                else:
                    raise ValueError(f"Unknown task: {task_name}")
            except (DocumentError, OllamaError, ValueError) as exc:
                self.worker_queue.put({"type": "error", "message": str(exc)})
            except Exception as exc:  # pragma: no cover - final safety net
                self.logger.exception("Unhandled background task failure")
                self.worker_queue.put({"type": "error", "message": "An unexpected error occurred while processing the request."})
            finally:
                if task_name != "status_check":
                    self.worker_queue.put({"type": "task_finished"})
                else:
                    self.worker_queue.put({"type": "task_finished"})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def browse_document(self) -> None:
        if self.is_busy:
            return
        file_path = filedialog.askopenfilename(
            title="Select document",
            filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf")],
        )
        if not file_path:
            return
        self._start_background_task("load_document", {"path": file_path})

    def clear_document(self) -> None:
        self.current_document = None
        self.document_name_var.set("No document loaded")
        self.document_path_var.set("Selected path: -")
        self.document_meta_var.set("Document details: Not available")
        self._append_message("system", "Document cleared from memory.")

    def new_chat(self) -> None:
        self.conversation_manager.clear()
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")
        self._append_message("assistant", "Started a new conversation. Ask a question or summarize the current document.")

    def ask_question(self) -> None:
        content = self.input_box.get("1.0", "end").strip()
        if self.input_box.get("1.0", "end").strip() == "Ask a question about your document...":
            content = ""
        if not content:
            self._append_message("system", "Please enter a question before sending.")
            return
        self.input_box.delete("1.0", "end")
        self.input_box.insert("1.0", "Ask a question about your document...")
        self.input_box.configure(fg="#7a8187")
        self._start_background_task("ask_question", {"question": content})

    def summarize_document(self) -> None:
        if self.is_busy:
            return
        self._start_background_task("summarize", {})

    def save_conversation(self) -> None:
        if not self.conversation_manager.messages:
            messagebox.showinfo("No conversation", "There is no conversation to save yet.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save conversation",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not file_path:
            return
        try:
            saved = self.conversation_manager.save_conversation(file_path, document_name=getattr(self.current_document, "file_name", None))
            messagebox.showinfo("Saved", f"Conversation saved to: {saved}")
        except OSError as exc:
            messagebox.showerror("Save failed", f"The conversation could not be saved: {exc}")

    def run(self) -> None:
        self.root.mainloop()
