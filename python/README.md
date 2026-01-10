# MLC-CLI with RAG Integration

This application integrates Retrieval-Augmented Generation (RAG) with MLC-LLM for enhanced context-aware responses and LaMP personalization benchmarks.

## Features

- **Service-Oriented Architecture**: Clean separation of concerns with dependency injection
- **Strict Typing**: Pydantic DTOs and Protocol interfaces for type safety
- **SQLite Database**: Uses SQLite with Prisma ORM for vector storage
- **RAG System**: Retrieves relevant context before generating LLM responses
- **LaMP Personalization**: Support for LaMP benchmark tasks (Citation, Movie Tagging, etc.)
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

2. Add sample documents or LaMP dataset to the database:
```bash
# Add generic documents for RAG
python -m src.add_documents

# Add LaMP dataset samples (requires LaMP data in LaMP/ directory)
python -m src.add_documents --lamp
```

## Project Structure

### New SOA Architecture (Recommended)
```
src/
├── main.py                    # Composition Root (entry point)
├── add_documents.py           # Document ingestion script
├── core/
│   ├── domain.py             # Pydantic DTOs (Document, Chunk, LaMPContext, etc.)
│   └── interfaces.py         # Protocol definitions (Embedder, VectorStore, etc.)
├── infrastructure/
│   ├── database.py           # PrismaConnection
│   ├── embedder.py           # LocalEmbedder (SentenceTransformers)
│   ├── vector_store.py       # PrismaVectorStore
│   ├── document_repository.py
│   ├── lamp_repository.py
│   └── llm_provider.py       # MLCLLMProvider
└── services/
    ├── rag_service.py        # RAG orchestration
    ├── lamp_service.py       # LaMP personalization
    └── document_service.py   # Document management
```

## Usage

### Running the Application

```bash
# Run with new SOA architecture
python -m src.main [OPTIONS]
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lamp` | flag | - | Enable LaMP personalization mode |
| `--sample-id` | string | - | LaMP sample ID to process (e.g., "010") |
| `--task` | int | - | LaMP task number (1-7) |
| `--split` | string | `validation` | Data split: `train`, `validation`, or `test` |
| `--variant` | string | `user_based` | Variant: `user_based` or `time_based` |
| `--top-k` | int | `5` | Number of profile documents to retrieve |
| `--query` | string | - | Custom query for generic RAG |
| `--list` | flag | - | List available LaMP tasks and samples |
| `--random` | flag | - | Select random LaMP task and sample |
| `--no-llm` | flag | - | Skip LLM inference (for testing retrieval only) |

### Usage Examples

#### Generic RAG Query
```bash
# Ask a question using RAG context
python -m src.main --query "What is machine learning?"

# Test retrieval without LLM
python -m src.main --query "What is AI?" --no-llm
```

#### LaMP Personalization

```bash
# List available LaMP tasks and samples
python -m src.main --list

# List samples for a specific task
python -m src.main --list --task 1

# Run specific LaMP sample
python -m src.main --lamp --sample-id "010" --task 1

# Run with custom top-k retrieval
python -m src.main --lamp --sample-id "010" --task 1 --top-k 10

# Run random LaMP sample
python -m src.main --random

# Test LaMP retrieval without LLM
python -m src.main --lamp --sample-id "010" --task 1 --no-llm
```

#### LaMP Task Examples

- **Task 1**: Citation Identification
- **Task 2**: Movie Tagging
- **Task 3**: Product Rating
- **Task 4**: News Headline Generation
- **Task 5**: Scholarly Title Generation
- **Task 6**: Tweet Paraphrasing
- **Task 7**: Email Subject Generation

## How It Works

### Generic RAG Flow
1. **Document Storage**: Documents are split into chunks and stored with embeddings
2. **Query Processing**: User queries are embedded using the same model
3. **Retrieval**: Most relevant chunks are retrieved using cosine similarity
4. **Context Building**: Retrieved chunks are formatted as context
5. **LLM Generation**: Context + query are sent to the LLM for enhanced responses

### LaMP Personalization Flow
1. **Sample Selection**: User selects a LaMP task and sample
2. **Profile Retrieval**: System retrieves user's historical data (publications, reviews, tweets, etc.)
3. **Embedding Search**: Most relevant profile items are found via semantic similarity
4. **Context Building**: Profile history is formatted with task-specific prompts
5. **Personalized Generation**: LLM generates response based on user's historical patterns

## Database Schema

### Generic RAG Tables

**Document**
- `id`: Unique identifier
- `content`: Full document text
- `metadata`: Optional JSON metadata
- `createdAt`: Creation timestamp

**Chunk**
- `id`: Unique identifier
- `documentId`: Reference to parent document
- `content`: Chunk text
- `embedding`: Serialized vector embedding
- `position`: Position in document
- `createdAt`: Creation timestamp

### LaMP Tables

**LaMPTask**
- `id`: Unique identifier
- `taskNumber`: Task number (1-7)
- `name`: Task name (e.g., "Citation_Identification")
- `description`: Task description

**LaMPSample**
- `id`: Unique identifier
- `sampleId`: Original dataset sample ID
- `taskId`: Reference to LaMPTask
- `input`: Task input/prompt
- `output`: Expected output (null for test set)
- `split`: Data split (train/validation/test)
- `variant`: Variant type (user_based/time_based)

**ProfileDocument**
- `id`: Unique identifier
- `profileId`: Original profile item ID
- `sampleId`: Reference to LaMPSample
- `title`: Optional title
- `content`: Profile text content
- `embedding`: Serialized vector embedding

## Architecture

### Dependency Injection

All services receive dependencies via constructor injection:

```python
# Infrastructure implementations
embedder = LocalEmbedder()
vector_store = PrismaVectorStore(prisma, embedder)
lamp_repo = PrismaLaMPRepository(prisma, embedder)

# Service layer (depends on interfaces, not concretions)
rag_service = RAGService(vector_store, embedder)
lamp_service = LaMPService(lamp_repo, embedder)
```

### Swapping Implementations

To use OpenAI embeddings instead of SentenceTransformers:

```python
# Create new implementation
class OpenAIEmbedder:
    def encode(self, text: str) -> list[float]: ...
    def encode_batch(self, texts: list[str]) -> list[list[float]]: ...
    def cosine_similarity(self, emb1: list[float], emb2: list[float]) -> float: ...

# Inject in composition root (src/main.py)
embedder = OpenAIEmbedder()  # Instead of LocalEmbedder()
```

No changes needed in service layer - it depends only on the `Embedder` protocol.
