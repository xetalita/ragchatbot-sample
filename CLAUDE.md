# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) chatbot system for course materials. It combines semantic search with AI generation to answer questions about educational content.

## Essential Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Development Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Set up environment
# Create .env file with:
ANTHROPIC_API_KEY=your_key_here
```

### Access Points
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture Overview

### Query Processing Pipeline

1. **Frontend Entry** (`frontend/script.js:45-96`)
   - Captures user input, sends POST to `/api/query`
   - Handles loading states and response display

2. **API Gateway** (`backend/app.py:56-74`)
   - FastAPI endpoint validates request
   - Manages session creation/retrieval
   - Routes to RAG system

3. **RAG Orchestration** (`backend/rag_system.py:102-140`)
   - Coordinates between AI, search tools, and session management
   - Maintains conversation history
   - Returns response with sources

4. **AI Decision Layer** (`backend/ai_generator.py:43-135`)
   - Claude analyzes query intent
   - Decides whether to search course content or respond directly
   - If search needed, executes tool calls

5. **Search Tool System** (`backend/search_tools.py`, `backend/vector_store.py`)
   - CourseSearchTool performs semantic search
   - VectorStore interfaces with ChromaDB
   - Returns relevant course chunks with metadata

### Key Components

**Backend Services:**
- `app.py`: FastAPI server with CORS, serves API and static files
- `rag_system.py`: Main orchestrator for RAG pipeline
- `ai_generator.py`: Claude API integration with tool support
- `vector_store.py`: ChromaDB wrapper for semantic search
- `search_tools.py`: Tool definitions for Claude's function calling
- `document_processor.py`: Chunks course documents for indexing
- `session_manager.py`: Tracks conversation history per session
- `config.py`: Central configuration (models, chunk sizes, API keys)

**Data Flow:**
1. User query → Frontend → API endpoint
2. API → RAG system → AI generator (with tools)
3. AI → Search tool → Vector store → ChromaDB
4. Results → AI → Final response generation
5. Response → Session update → Frontend display

### Important Configuration

Key settings in `backend/config.py`:
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation turns

### Document Processing

Course documents (`docs/course*.txt`) are:
1. Parsed to extract course metadata and lessons
2. Chunked into overlapping segments
3. Embedded using SentenceTransformer
4. Stored in ChromaDB collections:
   - `course_catalog`: Course titles/metadata
   - `course_content`: Searchable content chunks

### Tool-Based Search

The system uses Claude's function calling with `search_course_content` tool:
- Supports course name filtering (fuzzy matching)
- Lesson number filtering
- Semantic query search
- Returns formatted results with sources

### Session Management

Each user session:
- Gets unique session ID
- Maintains conversation history (last 2 exchanges)
- History provided as context for continuity
- Sessions stored in memory (not persistent)

## Development Notes

### Adding New Course Documents
Place `.txt`, `.pdf`, or `.docx` files in `docs/` directory. The system auto-loads on startup and skips duplicates based on course title.

### Modifying Search Behavior
- Adjust chunk size/overlap in `config.py`
- Modify search parameters in `vector_store.py`
- Update tool definitions in `search_tools.py`

### API Response Format
```json
{
  "answer": "AI-generated response",
  "sources": ["Course 1 - Lesson 2", ...],
  "session_id": "unique-session-id"
}
```

### Frontend Customization
- Chat UI: `frontend/index.html`
- Event handling: `frontend/script.js`
- Styling: `frontend/style.css`

### ChromaDB Storage
Vector database persisted at `backend/chroma_db/`. Delete this directory to reset the knowledge base.

## Windows Compatibility
Use Git Bash for running shell commands on Windows systems.
- always use uv to run the server, do not use pip directly
- make sure to use UV to manage all dependencies.
- use uv to run python files
