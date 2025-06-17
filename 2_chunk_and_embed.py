#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List

from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from transformers import AutoTokenizer

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "Qwen/Qwen3-Embedding-0.6B"
INPUT_PATH    = "parsed_docling_files"
DB_PATH       = "./docling_vectorstore.db"
COLLECTION    = "docling_demo"
DEVICE        = "cuda:1"
MAX_TOKENS    = 1_536


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _serialize_metadata(v):
    """Ensure lists & dicts are stringified for Milvus storage."""
    return json.dumps(v) if isinstance(v, (list, dict)) else v


def _make_doc_id(filename: str, chunk_index: int) -> str:
    """Stable identifier for a chunk; also used as the Milvus primary key."""
    return f"{filename}:{chunk_index}"


def _already_indexed(store: Milvus, doc_id: str) -> bool:
    """Quick point‑lookup to see whether this id is in the collection."""
    try:
        return bool(store.get(ids=[doc_id]))
    except Exception:
        # Milvus raises if the ID is not found → treat as "not yet indexed"
        return False


# ────────────────────────────────────────────────────────────────────────────────
# Document processing
# ────────────────────────────────────────────────────────────────────────────────

def _build_doc_items_info(chunk) -> list:
    """Extract the doc_items structure (same shape as the original script)."""
    if not (hasattr(chunk, "meta") and chunk.meta and chunk.meta.doc_items):
        return []

    info = []
    for it in chunk.meta.doc_items:
        item = {
            "label"   : getattr(it.label, "value", str(it.label)),
            "self_ref": it.self_ref,
        }
        prov = (it.prov or [None])[0]
        if prov:
            if getattr(prov, "bbox", None):
                item["bbox"] = {
                    "left"  : prov.bbox.l,
                    "top"   : prov.bbox.t,
                    "right" : prov.bbox.r,
                    "bottom": prov.bbox.b,
                }
            if hasattr(prov, "page_no"):
                item["page_no"] = prov.page_no
        info.append(item)
    return info


def chunk_documents(input_path: str, store: Milvus | None) -> List[Document]:
    """Return a list of *new* Documents that still need to be embedded."""
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL),
        max_tokens=MAX_TOKENS,
    )
    #https://docling-project.github.io/docling/concepts/chunking/#hybrid-chunker
    chunker   = HybridChunker(tokenizer=tokenizer)

    docs: List[Document] = []
    skipped = 0

    for filename in sorted(os.listdir(input_path)):
        if not filename.endswith(".json"):
            continue

        print(f"[•] {filename}")
        dl_doc = DoclingDocument.load_from_json(os.path.join(input_path, filename))

        for idx, chunk in enumerate(chunker.chunk(dl_doc)):
            doc_id = _make_doc_id(filename, idx)

            # Dedup check
            if store is not None and _already_indexed(store, doc_id):
                skipped += 1
                continue

            text = chunker.contextualize(chunk)
            metadata = {
                "doc_id"          : doc_id,
                "source"          : filename,
                "chunk_id"        : idx,
                "num_tokens"      : tokenizer.count_tokens(text=text),
                "doc_items_refs"  : [it.self_ref for it in chunk.meta.doc_items],
                "headings"        : chunk.meta.headings if chunk.meta and chunk.meta.headings else [],
                "original_filename": getattr(chunk.meta.origin, "filename", ""),
                "mimetype"        : getattr(chunk.meta.origin, "mimetype", ""),
                "doc_items"       : _build_doc_items_info(chunk),
            }
            metadata = {k: _serialize_metadata(v) for k, v in metadata.items()}

            docs.append(Document(page_content=text, metadata=metadata))

    print(f"  ↳ new chunks: {len(docs)}, skipped (already embedded): {skipped}")
    return docs


# ────────────────────────────────────────────────────────────────────────────────
# Vector‑store helpers
# ────────────────────────────────────────────────────────────────────────────────

def _embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
    )


def _new_store(docs: List[Document]) -> Milvus:
    """Create a fresh Milvus collection from `docs`."""
    db = Path(DB_PATH)
    db.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating collection '{COLLECTION}' with {len(docs)} chunks …")
    return Milvus.from_documents(
        documents       = docs,
        embedding       = _embedding_model(),
        collection_name = COLLECTION,
        connection_args = {"uri": str(db.absolute())},
        index_params    = {"index_type": "FLAT"},
        ids             = [d.metadata["doc_id"] for d in docs],
        drop_old        = True,
    )


def _load_store() -> Milvus:
    """Load existing Milvus collection with embedding function."""
    # Try different parameter names based on langchain_milvus version
    embedding_model = _embedding_model()
    
    try:
        # Try with embedding_function parameter
        return Milvus(
            embedding_function = embedding_model,
            collection_name = COLLECTION,
            connection_args = {"uri": str(Path(DB_PATH).absolute())},
        )
    except TypeError:
        try:
            # Try with embeddings parameter  
            return Milvus(
                embeddings = embedding_model,
                collection_name = COLLECTION,
                connection_args = {"uri": str(Path(DB_PATH).absolute())},
            )
        except TypeError:
            # Fall back to basic constructor and set embedding after
            store = Milvus(
                collection_name = COLLECTION,
                connection_args = {"uri": str(Path(DB_PATH).absolute())},
            )
            # Try to set the embedding function manually
            if hasattr(store, 'embedding_func'):
                store.embedding_func = embedding_model
            elif hasattr(store, '_embedding'):
                store._embedding = embedding_model
            return store


def _upsert(store: Milvus, docs: List[Document]):
    if not docs:
        print("Nothing new to embed – collection already up‑to‑date ✅")
        return

    print(f"Adding {len(docs)} new chunks …")
    store.add_documents(
        documents = docs,
        ids       = [d.metadata["doc_id"] for d in docs],
    )


# ────────────────────────────────────────────────────────────────────────────────
# Demo similarity search
# ────────────────────────────────────────────────────────────────────────────────

def _search_demo(store: Milvus):
    queries = [
        "artificial intelligence models",
        "table structure recognition",
        "document processing pipeline",
    ]
    for q in queries:
        print(f"\n🔍 {q!r}")
        for i, doc in enumerate(store.similarity_search(q, k=2), 1):
            snippet = doc.page_content[:96].replace("\n", " ") + "…"
            print(f"{i:>2}. {snippet:<100} ({doc.metadata.get('source')})")


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    input_dir = Path(INPUT_PATH)
    if not input_dir.exists():
        raise SystemExit(f"No input directory: {input_dir}")

    store = None
    if Path(DB_PATH).exists():
        try:
            store = _load_store()
            print(f"Loaded existing collection '{COLLECTION}'")
        except Exception as exc:
            print(f"Could not load existing store ({exc}). Re‑creating …")

    new_docs = chunk_documents(str(input_dir), store)

    if store:
        _upsert(store, new_docs)
    else:
        store = _new_store(new_docs)

    _search_demo(store)
    print("\nDone ✅")


if __name__ == "__main__":
    main()
