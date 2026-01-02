# MLC-CLI with RAG Integration

This application integrates Retrieval-Augmented Generation (RAG) with MLC-LLM for enhanced context-aware responses.

## Features

- **SQLite Database**: Uses SQLite with sqlite-vec extension for vector storage
- **Prisma ORM**: Type-safe database access with Prisma for Python
- **RAG System**: Retrieves relevant context before generating LLM responses
- **Embedding Model**: Uses sentence-transformers for text embeddings
- **Async Support**: Full async/await support for efficient operations

## Setup

1. Install dependencies and initialize database:
```bash
bash setup.sh
```

Or manually:
```bash
pip install -r requirements.txt
prisma generate
prisma db push
```

2. Add sample documents to the RAG database:
```bash
python add_documents.py
```

3. Run the main application:
```bash
python main.py
```

## Project Structure

- `main.py` - Main application with RAG-enhanced LLM responses
- `schema.prisma` - Prisma schema for database models
- `db_init.py` - Database initialization utilities
- `embeddings.py` - Embedding model wrapper
- `rag.py` - RAG system implementation
- `add_documents.py` - Utility to add documents to the database
- `requirements.txt` - Python dependencies

## How It Works

1. **Document Storage**: Documents are split into chunks and stored with embeddings
2. **Query Processing**: User queries are embedded using the same model
3. **Retrieval**: Most relevant chunks are retrieved using cosine similarity
4. **Context Building**: Retrieved chunks are formatted as context
5. **LLM Generation**: Context + query are sent to the LLM for enhanced responses

## Database Schema

### Document
- `id`: Unique identifier
- `content`: Full document text
- `metadata`: Optional JSON metadata
- `createdAt`: Creation timestamp

### Chunk
- `id`: Unique identifier
- `documentId`: Reference to parent document
- `content`: Chunk text
- `embedding`: Serialized vector embedding
- `position`: Position in document
- `createdAt`: Creation timestamp

## Customization

### Change Embedding Model
Edit `embeddings.py` and modify the model name:
```python
EmbeddingModel(model_name="all-MiniLM-L6-v2")
```

### Adjust Chunk Size
When adding documents:
```python
await rag_system.add_document(content, metadata, chunk_size=500)
```

### Change Top-K Retrieval
In `main.py`:
```python
context = await rag_system.build_context(user_query, top_k=3)
```

## API Usage

### Add Document
```python
doc_id = await rag_system.add_document(
    content="Your document text here",
    metadata={"source": "example", "topic": "ai"},
    chunk_size=500
)
```

### Retrieve Relevant Chunks
```python
chunks = await rag_system.retrieve_relevant_chunks(
    query="What is AI?",
    top_k=5
)
```

### Build Context
```python
context = await rag_system.build_context(
    query="What is AI?",
    top_k=3
)
```

### List Documents
```python
documents = await rag_system.list_documents()
```

### Delete Document
```python
success = await rag_system.delete_document(document_id)
```
