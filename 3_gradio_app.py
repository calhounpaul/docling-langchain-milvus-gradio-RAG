import os
import json
import torch
import gradio as gr
from pathlib import Path
from typing import List, Tuple
import re

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./docling_vectorstore.db"
LLM_MODEL = "Qwen/Qwen3-1.7B"
COLLECTION = "docling_demo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vectorstore = None
llm_pipeline = None
tokenizer = None

def init_components():
    global vectorstore, llm_pipeline, tokenizer
    
    db_path = Path(DB_PATH)
    if not db_path.exists():
        raise FileNotFoundError(f"Vector store not found: {db_path}")
    
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={'device': DEVICE})
    
    try:
        vectorstore = Milvus(
            embedding_function=embedding,
            collection_name=COLLECTION,
            connection_args={"uri": str(db_path.absolute())},
            auto_id=True
        )
    except TypeError:
        vectorstore = Milvus(
            embedding_function=embedding,
            collection_name=COLLECTION,
            connection_args={"uri": str(db_path.absolute())}
        )
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
    )
    
    print(f"Components loaded: {DEVICE}")

def get_chunks(query: str) -> Tuple[List[str], List[str]]:
    if vectorstore is None:
        raise RuntimeError("Vector store not initialized")
    
    docs = vectorstore.similarity_search(query, k=5)
    chunks = [doc.page_content.strip() for doc in docs]
    sources = set()
    
    for doc in docs:
        source = doc.metadata.get('source') or doc.metadata.get('original_filename')
        if source:
            sources.add(source)
    
    return chunks, list(sources)

def format_prompt(query: str, chunks: List[str]) -> str:
    parts = ["Use the following snippets of text to answer the user query:"]
    
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"\nSnippet {i}:")
        parts.append(chunk)
    
    parts.append(f"\nUse the above snippets of text to answer the following user query: {query}")
    return "\n".join(parts)

def generate_response(query: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
    if llm_pipeline is None:
        raise RuntimeError("LLM not initialized")
    
    try:
        chunks, sources = get_chunks(query)
        rag_prompt = format_prompt(query, chunks)
        
        messages = []
        for user_msg, assistant_msg in history[-2:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": rag_prompt})
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        output = llm_pipeline(
            text,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            return_full_text=False
        )
        
        response = output[0]['generated_text']
        
        # Clean response
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        response = response.strip()
        
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1].strip()
        
        if not response:
            response = "I couldn't generate a proper response. Please try rephrasing your question."
        
        source_info = f"Sources: {', '.join(sources)}" if sources else ""
        return response, source_info
        
    except Exception as e:
        return f"Error: {str(e)}", ""

def create_interface():
    def user_submit(message, history):
        return "", history + [[message, None]]
    
    def bot_response(history, show_sources, show_chunks):
        if not history or history[-1][1] is not None:
            return history, gr.update(visible=False), gr.update(visible=False)
        
        user_message = history[-1][0]
        
        try:
            response, source_info = generate_response(user_message, history[:-1])
            
            if show_sources and source_info:
                response += f"\n\n{source_info}"
            
            history[-1][1] = response
            
            source_display = ""
            chunks_display = ""
            
            if show_sources or show_chunks:
                try:
                    chunks, sources = get_chunks(user_message)
                    if show_sources:
                        source_display = f"Sources: {', '.join(sources)}"
                    if show_chunks:
                        chunks_display = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
                except Exception as e:
                    source_display = f"Error: {str(e)}"
                    chunks_display = f"Error: {str(e)}"
            
            return (
                history, 
                gr.update(visible=show_sources, value=source_display),
                gr.update(visible=show_chunks, value=chunks_display)
            )
            
        except Exception as e:
            history[-1][1] = f"Error: {str(e)}"
            return (
                history,
                gr.update(visible=True, value=f"Error: {str(e)}"),
                gr.update(visible=False)
            )
    
    with gr.Blocks(title="RAG Document Q&A") as demo:
        gr.Markdown("# RAG Document Q&A\nRetrieval-Augmented Generation chatbot for research papers.")
        
        chatbot = gr.Chatbot(height=600, label="Chat")
        
        with gr.Row():
            msg = gr.Textbox(label="Question", placeholder="Ask about the research papers...", lines=2, scale=4)
            submit = gr.Button("Send", variant="primary", scale=1)
        
        clear = gr.Button("Clear", variant="secondary")
        
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                show_sources = gr.Checkbox(value=True, label="Show sources")
                show_chunks = gr.Checkbox(value=False, label="Show chunks")
        
        source_display = gr.Textbox(label="Sources", lines=2, visible=True, interactive=False)
        chunks_display = gr.Textbox(label="Chunks", lines=10, visible=False, interactive=False)
        
        gr.Examples(
            examples=[
                "What are the main machine learning techniques?",
                "How does XGBoost work?",
                "What is human-in-the-loop machine learning?",
                "Explain gradient descent algorithms",
                "What are AI challenges?",
                "How is deep learning used in biomedical domains?",
            ],
            inputs=msg,
        )
        
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, show_sources, show_chunks], [chatbot, source_display, chunks_display]
        )
        
        submit.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, show_sources, show_chunks], [chatbot, source_display, chunks_display]
        )
        
        clear.click(lambda: (None, gr.update(visible=False), gr.update(visible=False)), 
                   outputs=[chatbot, source_display, chunks_display])
        
        show_sources.change(lambda x: gr.update(visible=x), [show_sources], [source_display])
        show_chunks.change(lambda x: gr.update(visible=x), [show_chunks], [chunks_display])
    
    return demo

def main():
    print(f"RAG Chatbot | DB: {DB_PATH} | Model: {LLM_MODEL} | Device: {DEVICE}")
    
    try:
        init_components()
        demo = create_interface()
        
        print("Starting at http://localhost:7860")
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure vector database exists and sufficient GPU memory available")
        raise

if __name__ == "__main__":
    main()
