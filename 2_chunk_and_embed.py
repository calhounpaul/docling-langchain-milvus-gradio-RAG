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

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INPUT_PATH = "parsed_docling_files"
DB_PATH = "./docling_vectorstore.db"

def serialize_metadata(value):
    return json.dumps(value) if isinstance(value, (list, dict)) else value

def process_documents(input_path: str) -> List[Document]:
    tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL))
    chunker = HybridChunker(tokenizer=tokenizer)
    documents = []

    for filename in os.listdir(input_path):
        if not filename.endswith(".json"):
            continue
            
        print(f"Processing {filename}")
        doc = DoclingDocument.load_from_json(os.path.join(input_path, filename))
        chunks = list(chunker.chunk(dl_doc=doc))

        for i, chunk in enumerate(chunks):
            text = chunker.contextualize(chunk=chunk)
            
            metadata = {
                "source": filename,
                "chunk_id": i,
                "num_tokens": tokenizer.count_tokens(text=text),
                "doc_items_refs": json.dumps([it.self_ref for it in chunk.meta.doc_items]),
                "headings": json.dumps([]),
                "original_filename": "",
                "mimetype": "",
                "doc_items": json.dumps([]),
            }

            if hasattr(chunk, 'meta') and chunk.meta:
                if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                    metadata["headings"] = json.dumps(chunk.meta.headings)
                if hasattr(chunk.meta, 'origin') and chunk.meta.origin:
                    origin = chunk.meta.origin
                    if hasattr(origin, 'filename'):
                        metadata["original_filename"] = origin.filename
                    if hasattr(origin, 'mimetype'):
                        metadata["mimetype"] = origin.mimetype

                doc_items_info = []
                for doc_item in chunk.meta.doc_items:
                    item_info = {
                        "label": doc_item.label.value if hasattr(doc_item.label, 'value') else str(doc_item.label),
                        "self_ref": doc_item.self_ref
                    }
                    if hasattr(doc_item, 'prov') and doc_item.prov:
                        prov = doc_item.prov[0]
                        if hasattr(prov, 'bbox') and prov.bbox:
                            item_info["bbox"] = {
                                "left": prov.bbox.l, "top": prov.bbox.t,
                                "right": prov.bbox.r, "bottom": prov.bbox.b
                            }
                        if hasattr(prov, 'page_no'):
                            item_info["page_no"] = prov.page_no
                    doc_items_info.append(item_info)
                metadata["doc_items"] = json.dumps(doc_items_info)

            for key, value in metadata.items():
                metadata[key] = serialize_metadata(value)

            documents.append(Document(page_content=text, metadata=metadata))

    print(f"Processed {len(documents)} chunks")
    return documents

def create_vector_store(documents: List[Document]) -> Milvus:
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={'device': 'cuda:1'})
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    vectorstore = Milvus.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name="docling_demo",
        connection_args={"uri": str(db_path.absolute())},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )

    print(f"Vector store created: {len(documents)} docs -> {db_path}")
    return vectorstore

def load_vector_store() -> Milvus:
    db_path = Path(DB_PATH)
    if not db_path.exists():
        raise FileNotFoundError(f"No vector store at {db_path}")

    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={'device': 'cuda:1'})
    return Milvus(
        embedding=embedding,
        collection_name="docling_demo",
        connection_args={"uri": str(db_path.absolute())},
    )

def search_docs(vectorstore: Milvus, query: str, k: int = 3) -> List[Document]:
    results = vectorstore.similarity_search(query, k=k)
    print(f"Found {len(results)} results for: {query}")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}... (from {doc.metadata.get('source', 'unknown')})")
    return results

def main():
    db_path = Path(DB_PATH)
    
    if db_path.exists():
        choice = input("Vector store exists. (L)oad or (R)ecreate? ").upper()
        if choice == 'L':
            try:
                vectorstore = load_vector_store()
                print("Loaded existing vector store")
            except Exception as e:
                print(f"Load failed: {e}")
                vectorstore = None
        else:
            vectorstore = None
    else:
        vectorstore = None

    if vectorstore is None:
        if not os.path.exists(INPUT_PATH):
            print(f"No input documents at {INPUT_PATH}")
            return
        
        documents = process_documents(INPUT_PATH)
        if not documents:
            print("No documents processed")
            return
        
        vectorstore = create_vector_store(documents)

    # Test searches
    test_queries = [
        "artificial intelligence models",
        "table structure recognition", 
        "document processing pipeline"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        search_docs(vectorstore, query, k=2)

    print("Done")

if __name__ == "__main__":
    main()
