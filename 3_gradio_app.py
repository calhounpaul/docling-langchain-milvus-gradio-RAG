import os
import re
import threading
from pathlib import Path
from typing import List, Tuple

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# ------------------------------
# CONFIGURATION CONSTANTS
# ------------------------------
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DB_PATH = "./docling_vectorstore.db"
LLM_MODEL = "Qwen/Qwen3-4B"
COLLECTION = "docling_demo"
DEVICE = "cuda"

# ------------------------------
# GLOBALS (loaded lazily)
# ------------------------------
vectorstore = None
model = None
tokenizer = None

# ------------------------------
# INITIALISATION HELPERS
# ------------------------------

def init_components():
    """Load everything we need exactly once (embedding, vectorstore, model)."""
    global vectorstore, model, tokenizer

    if vectorstore and model and tokenizer:
        return  # already initialised

    db_path = Path(DB_PATH)
    if not db_path.exists():
        raise FileNotFoundError(f"Vector store not found: {db_path.absolute()}")

    embedding = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda:1"},
    )

    try:
        vectorstore = Milvus(
            embedding_function=embedding,
            collection_name=COLLECTION,
            connection_args={"uri": str(db_path.absolute())},
            auto_id=True,
        )
    except TypeError:
        # `auto_id` only exists for milvus>=2.3 – fall back silently
        vectorstore = Milvus(
            embedding_function=embedding,
            collection_name=COLLECTION,
            connection_args={"uri": str(db_path.absolute())},
        )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map={"":"cuda:0"},
        trust_remote_code=True,
    )

    print(f"Components loaded on {DEVICE}")

# ------------------------------
# RAG HELPERS
# ------------------------------

def get_chunks(query: str) -> Tuple[List[str], List[str]]:
    """Retrieve the most relevant chunks for a query and return chunks + source names."""
    if vectorstore is None:
        raise RuntimeError("Vector store not initialised – call init_components() first")

    docs = vectorstore.similarity_search(query, k=5)
    chunks = [doc.page_content.strip() for doc in docs]
    sources: List[str] = []

    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("original_filename")
        if src and src not in sources:
            sources.append(src)

    return chunks, sources


def format_prompt(query: str, chunks: List[str]) -> str:
    """Create a chat-style prompt with embedded snippets for the LLM."""
    parts = [
        "Use the following snippets of text to answer the user query:",
    ]
    for idx, chunk in enumerate(chunks, 1):
        parts.append(f"\nSnippet {idx}:\n{chunk}")
    parts.append(f"\nUse the above snippets of text to answer the following user query: {query}")
    return "\n".join(parts)

# ------------------------------
# THINK‑TAG FILTERING
# ------------------------------

def _strip_think_tags(text: str, state: dict) -> str:
    """Incrementally strip <think>...</think> blocks from a streamed chunk."""
    output = []
    i = 0
    while i < len(text):
        if not state["in_think"] and text.startswith("<think>", i):
            state["in_think"] = True
            i += 7  # len("<think>")
            continue
        if state["in_think"]:
            end = text.find("</think>", i)
            if end == -1:
                # we haven't closed the tag yet – swallow everything until next chunk
                return ""  # produce nothing this round
            else:
                state["in_think"] = False
                i = end + 8  # len("</think>")
                continue
        output.append(text[i])
        i += 1
    return "".join(output)

# ------------------------------
# STREAMING GENERATOR
# ------------------------------

def stream_response(query: str, history: List[Tuple[str, str]]):
    """Yield partial assistant messages for Gradio to consume."""
    if model is None or tokenizer is None:
        raise RuntimeError("LLM not initialised – call init_components() first")

    # --- prepare RAG prompt ---
    chunks, sources = get_chunks(query)
    rag_prompt = format_prompt(query, chunks)

    # Re‑build chat history with user & assistant turns (2 latest for brevity)
    messages = []
    for user_msg, assistant_msg in history[-2:]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": rag_prompt})

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --- kick off background generation thread ---
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )

    inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
    ).to(model.device)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024*4,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # --- stream tokens back, hiding <think> … </think> ---
    response_so_far = ""
    tag_state = {"in_think": False}
    for new_text in streamer:
        clean_piece = _strip_think_tags(new_text, tag_state)
        if clean_piece:
            response_so_far += clean_piece.replace("<|im_end|>","")
            yield response_so_far

    # Once generation is done, append sources if we have any
    if sources:
        #response_so_far += f"\n\nSources:\n - {'\n - '.join(["["+"_".join(s.split("_")[1:])[:-5]+"](https://arxiv.org/abs/" + s.split("_")[0]+")" for s in sources])}"
        response_so_far += f"\n\nSources:\n - {'\n - '.join(["["+s.split("_")[0]+"](https://arxiv.org/abs/" + s.split("_")[0]+")" for s in sources])}"
        yield response_so_far

# ------------------------------
# GRADIO INTERFACE
# ------------------------------

def create_interface():
    def user_submit(message, history):
        """Append user message (placeholder assistant) then return."""
        return "", history + [[message, ""]]

    def bot_response(history, show_sources, show_chunks):
        """Generator function compatible with Gradio Chatbot streaming."""
        if not history or history[-1][1] not in (None, ""):
            # nothing to do
            yield history, gr.update(), gr.update()
            return

        user_msg = history[-1][0]
        # start streaming
        gen = stream_response(user_msg, history[:-1])

        for partial_answer in gen:
            history[-1][1] = partial_answer
            yield history, gr.update(visible=False), gr.update(visible=False)

    with gr.Blocks(title="RAG Document Q&A (Streaming)") as demo:
        gr.Markdown("""# RAG Document Q&A (Streaming)\nEnjoy real‑time answers with source citations.""")

        chatbot = gr.Chatbot(height=300, label="Chat", show_copy_button=True)

        with gr.Row():
            msg = gr.Textbox(placeholder="Ask about the research papers…", lines=2, scale=4)
            submit = gr.Button("Send", variant="primary")

        clear = gr.Button("Clear")

        with gr.Accordion("Settings", open=False):
            with gr.Row():
                show_sources = gr.Checkbox(value=True, label="Show sources (end)")
                show_chunks = gr.Checkbox(value=False, label="Show chunks (dev mode)")

        source_display = gr.Textbox(label="Sources", visible=False, interactive=False)
        chunks_display = gr.Textbox(label="Chunks", visible=False, interactive=False)

        # wiring
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response,
            [chatbot, show_sources, show_chunks],
            [chatbot, source_display, chunks_display],
        )
        submit.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response,
            [chatbot, show_sources, show_chunks],
            [chatbot, source_display, chunks_display],
        )

        clear.click(lambda: ([], gr.update(visible=False), gr.update(visible=False)),
                    outputs=[chatbot, source_display, chunks_display])

        # toggle visibility on setting change
        show_sources.change(lambda x: gr.update(visible=x), show_sources, source_display)
        show_chunks.change(lambda x: gr.update(visible=x), show_chunks, chunks_display)

    return demo

# ------------------------------
# MAIN ENTRY POINT
# ------------------------------

def main():
    print(f"RAG Chatbot | DB: {DB_PATH} | Model: {LLM_MODEL} | Device: {DEVICE}")
    init_components()
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)

if __name__ == "__main__":
    main()
