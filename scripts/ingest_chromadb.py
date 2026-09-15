"""
scripts/ingest_chromadb.py

Ingests pipeline_final.json chunks into ChromaDB with D2 representation.
Run from project root: python scripts/ingest_chromadb.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from openai import OpenAI
from config.settings import get_settings

s = get_settings()
client = OpenAI(api_key=s.openai_api_key)
chroma = chromadb.PersistentClient(path=str(s.chroma_persist_dir))

with open('data/chunks/pipeline_final.json') as f:
    chunks = json.load(f)

print(f'Total chunks: {len(chunks)}')

def build_d2_text(chunk):
    parts = [p for p in [
        chunk.get('document_title', ''),
        chunk.get('section_title', ''),
        chunk.get('text', ''),
    ] if p and p.strip()]
    return '\n'.join(parts)

def embed_batch(texts):
    resp = client.embeddings.create(
        model=s.openai_embedding_model,
        input=texts,
    )
    return [r.embedding for r in resp.data]

by_namespace = {}
for chunk in chunks:
    ns = chunk.get('client', 'global')
    if ns not in by_namespace:
        by_namespace[ns] = []
    by_namespace[ns].append(chunk)

print(f'Namespaces: {sorted(by_namespace.keys())}')

total_ingested = 0
batch_size = 50

for namespace, ns_chunks in sorted(by_namespace.items()):
    collection_name = f'kira_{namespace}'
    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'}
    )
    print(f'\nIngesting {namespace} ({len(ns_chunks)} chunks) → {collection_name}')

    for i in range(0, len(ns_chunks), batch_size):
        batch = ns_chunks[i:i + batch_size]
        texts = [build_d2_text(c) for c in batch]
        embeddings = embed_batch(texts)

        ids = [c['chunk_id'] for c in batch]
        metadatas = [{
            'chunk_text':     c['text'],
            'window_context': c.get('window_context', c['text']),
            'source_file':    c['source_file'],
            'doc_type':       c['doc_type'],
            'client':         c.get('client', 'global'),
            'section_title':  c.get('section_title', ''),
            'document_title': c.get('document_title', ''),
            'chunk_strategy': c.get('chunk_strategy', ''),
            'embedding_repr': 'D2',
        } for c in batch]
        documents = [c['text'] for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        total_ingested += len(batch)
        print(f'  Batch {i//batch_size + 1}: {len(batch)} vectors')
        time.sleep(0.3)

print(f'\nVerification:')
chroma2 = chromadb.PersistentClient(path=str(s.chroma_persist_dir))
for c in chroma2.list_collections():
    print(f'  {c.name}: {c.count()} vectors')

print(f'\nDone. {total_ingested} vectors ingested to ChromaDB with D2 representation.')
