# Hyperthymesia Development Notes

## Project Overview
Building **Hyperthymesia** - a local-first AI search and Q&A tool for developers.

**Vision**: Universal search for everything (files, emails, browser history, Slack, etc.)
**Current Focus**: Code search MVP for developers
**Goal**: Apply to Y Combinator

## The Pitch
"The only AI assistant that's truly private - runs entirely on your device using small language models. While competitors like Rewind.ai send your data to the cloud, we process everything locally. Developers are our wedge - 28M globally, already pay for AI tools, value privacy."

## What's Built (Week 1-2)

### ✅ Core Features Working
1. **Hybrid Search Engine**
   - SQLite FTS5 (keyword search with BM25 ranking)
   - ChromaDB + sentence-transformers (semantic embeddings)
   - RRF (Reciprocal Rank Fusion) for combining results
   - Filename search with 3x boost
   - Query complexity detection

2. **File Type Support**
   - Code files: .py, .js, .ts, .java, .go, .rs, .cpp, .c, etc.
   - Text files: .txt, .md, .json, .csv, .yaml, .xml
   - PDFs: Full text extraction
   - Symbol extraction: Functions, classes, methods

3. **Developer Features**
   - Gitignore support (auto-filters node_modules, .venv, etc.)
   - Smart dependency handling (indexes docs only, not code)
   - Code-aware chunking (preserves functions/classes)
   - Pretty CLI with syntax highlighting

4. **Intelligence Layer**
   - Query rewriting for complex queries (uses Gemini API)
   - Simple keyword extraction for RAG (no API needed)
   - Query complexity analysis (routes to LLM only when needed)

5. **RAG Q&A System** (85% complete)
   - Document chunking by file type
   - Context retrieval with token limits
   - Local LLM integration via Ollama (Llama 3.2)
   - Cloud fallback (BYOK - users bring own API keys)
   - Answer with source citations

### CLI Commands
```bash
# Indexing
hyperthymesia index add <path>           # Index files
hyperthymesia index list                  # List indexed locations
hyperthymesia index refresh              # Re-index everything
hyperthymesia index stats --detailed     # Show statistics

# Search
hyperthymesia search "query"             # Hybrid search
hyperthymesia search "query" --verbose   # Show search strategy
hyperthymesia search "query" --file-type py  # Filter by type

# Q&A (RAG)
hyperthymesia ask "question"             # Ask about your code
hyperthymesia ask "question" --verbose   # Show retrieval process
hyperthymesia ask "question" --use-cloud # Use cloud LLM (better quality)
```

## Current Status (Last Session)

### What Works Perfectly ✅
- Indexing speed and accuracy
- Search finds correct files
- Gitignore filtering
- LLM integration (Ollama works great)
- CLI user experience
- Query rewriting for search

### Known Issue ⚠️
**RAG retrieval scoring needs fix:**
- **Symptom**: `ask "what database do we use?"` returns node_modules docs instead of user's config/db.js
- **Root cause**: RAG doesn't filter/boost user code over dependencies
- **Search works**: `search "database"` finds db.js correctly
- **Solution**: Add dependency filter in `core/rag_retriever.py` line ~45

### Fix to Apply
In `core/rag_retriever.py`, in `retrieve_context()` method after getting search results:
```python
# 2. FILTER OUT DEPENDENCIES
dep_dirs = {'node_modules', '__pycache__', '.venv', 'venv', 'env', 
            'site-packages', 'vendor', 'target', 'dist'}

def is_user_code(result):
    if 'metadata' in result:
        path = result.get('metadata', {}).get('path', '')
    else:
        path = result.get('path', '')
    return not any(dep in path for dep in dep_dirs)

semantic_results = [r for r in semantic_results if is_user_code(r)]
keyword_results = [r for r in keyword_results if is_user_code(r)]

# 3. BOOST USER CODE
for result in semantic_results + keyword_results:
    if 'score' in result:
        result['score'] *= 2.0
```

## Tech Stack

### Core
- **Language**: Python 3.11+
- **Search**: SQLite FTS5 + ChromaDB + sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama (Llama 3.2 3B/11B) with cloud BYOK fallback
- **CLI**: Click framework
- **Embeddings**: sentence-transformers (local, 400MB model)

### Dependencies
```
click>=8.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
PyPDF2>=3.0.0
platformdirs>=3.0.0
tqdm>=4.65.0
requests>=2.31.0

# Optional (user installs as needed):
# google-generativeai  # For query rewriting
# openai              # For cloud Q&A
# ollama              # Local LLM (install via: https://ollama.ai)
```

## Architecture

### Project Structure
```
hyperthymesia_cli/
├── __main__.py              # Entry point
├── cli/
│   ├── commands.py          # CLI commands (index, search, ask)
│   └── formatters.py        # Pretty output
├── core/
│   ├── indexer.py           # File indexing
│   ├── searcher.py          # Hybrid search (keyword + semantic)
│   ├── intelligent_searcher.py  # Query analysis & rewriting
│   ├── chunker.py           # Document chunking for RAG
│   ├── rag_retriever.py     # Context retrieval ⚠️ needs fix
│   ├── local_llm.py         # Ollama integration
│   ├── query_analyzer.py    # Query complexity detection
│   └── llm_helper.py        # LLM query rewriting (Gemini)
├── storage/
│   ├── db.py                # SQLite with FTS5
│   ├── vector_store.py      # ChromaDB wrapper
│   └── models.py            # Data models
├── parsers/
│   ├── text.py              # Text file parser
│   ├── pdf.py               # PDF parser
│   └── code.py              # Code parser with symbols
├── utils/
│   ├── gitignore.py         # Dependency filtering
│   ├── editor.py            # Editor detection (for --open)
│   ├── clipboard.py         # Clipboard utilities
│   └── logger.py            # Logging
└── config/                  # Configuration (future)
```

### Data Flow

**Indexing:**
```
Files → Gitignore Filter → Parser → SQLite (keyword) + ChromaDB (embeddings)
```

**Search:**
```
Query → Complexity Analysis → [Optional LLM Rewrite] → Hybrid Search → RRF → Results
```

**Q&A (RAG):**
```
Question → Search Documents → Chunk → Retrieve Context → Local LLM → Answer + Sources
```

## Key Design Decisions

### Why Local-First?
1. **Privacy**: No data leaves user's computer
2. **Cost**: No per-query API fees (sustainable free tier)
3. **Speed**: No network latency
4. **Offline**: Works on planes, secure environments
5. **Differentiation**: Unique vs Rewind.ai, Cursor, Perplexity

### Why Developers First?
1. Early adopters who understand technical value
2. Willing to pay ($20-50/month like Cursor)
3. Strong word-of-mouth in dev community
4. Harder use case (if we nail this, everyone else is easier)
5. Proven wedge strategy (Dropbox, Slack, Notion did this)

### Why Ollama?
1. Easiest local LLM setup (vs llama-cpp-python)
2. Good model selection (Llama 3.2 3B/11B)
3. Auto-updates and model management
4. Active community
5. Works well enough (~70% of GPT-4 quality)

### Why Hybrid Search?
1. Keyword catches exact matches (function names, file names)
2. Semantic catches concepts ("authentication" finds JWT code)
3. RRF balances both without tuning weights
4. Better than either alone

## Database Location
- **macOS**: `~/Library/Application Support/hyperthymesia/`
- **Linux**: `~/.local/share/hyperthymesia/`
- **Windows**: `C:\Users\<user>\AppData\Local\hyperthymesia\`

**Delete to reset**: `rm -rf ~/Library/Application\ Support/hyperthymesia/`

## Testing Commands
```bash
# Quick test flow
python __main__.py index add ~/your-project
python __main__.py search "authentication"
python __main__.py ask "how does auth work?"

# Debug mode
python __main__.py search "query" --verbose
python __main__.py ask "question" --verbose

# Quality comparison
python __main__.py ask "question"              # Local LLM
python __main__.py ask "question" --use-cloud  # Cloud LLM
```

## Next Steps (Priority Order)

### Immediate (This Week)
1. ✅ Fix RAG dependency filtering (30 min)
2. ✅ Test with 10 real questions (1 hour)
3. ✅ Evaluate quality: Is local LLM good enough? (critical decision!)
4. ✅ Polish error messages and UX (1 hour)

### Week 2
5. Ship to 5-10 beta developers
6. Get brutally honest feedback
7. Iterate based on what they actually use
8. Decide: Ship with local LLM or pivot to BYOK?

### Week 3-4 (Pre-YC)
9. Demo video (2-3 minutes showing the magic)
10. Landing page with value prop
11. Get 50+ users or strong testimonials
12. Measure: Daily active usage, searches per user
13. Polish YC application

## Success Metrics

**Week 1 (MVP Validation):**
- [ ] Can answer 7/10 questions correctly
- [ ] Speed <5s per query
- [ ] Works on real codebases (10k+ files)
- [ ] No crashes

**Week 2-4 (User Validation):**
- [ ] 10+ active beta users
- [ ] 5+ strong testimonials
- [ ] Users search 5+ times per day
- [ ] 3+ users willing to pay

**Month 2 (YC Application):**
- [ ] 100+ users
- [ ] 10+ paying users (even $5/month validates)
- [ ] Clear use cases documented
- [ ] Demo that wows

## Competitive Landscape

| Competitor | Weakness | Our Advantage |
|-----------|----------|---------------|
| **Rewind.ai** | Cloud-based, $75M raised | Fully local, private |
| **Cursor** | Code editor only | Cross-app (future) |
| **GitHub Copilot** | Code generation, not search | Search + Q&A |
| **Perplexity** | Internet search | Personal data |
| **Apple Intelligence** | Closed, cloud hybrid | Open, fully local |

## YC Pitch Elements

**One-liner**: "Rewind.ai but actually private - fully local AI memory for developers"

**Problem**: Developers spend 30% of time searching for code they wrote, docs they read, or decisions they made

**Solution**: Local AI that indexes everything, searches semantically, answers questions - all on your device

**Why now**: 
- SLMs (Llama 3.2) are finally good enough (~70% of GPT-4)
- Developers want privacy (GitHub Copilot concerns)
- On-device AI is trending (Apple Intelligence)

**Traction**: 
- [X] active developers
- [X] searches per day
- [X] testimonials

**Market**: 28M developers globally → 1B+ knowledge workers

**Why us**: [Add your background - why you're the right person to build this]

## Open Questions

1. **Quality Threshold**: Is 70% of GPT-4 good enough? Or do we need BYOK for serious adoption?
2. **Pricing**: Free with local LLM + $10/month pro? Or all BYOK?
3. **Distribution**: How do we get first 1000 developers?
4. **Expansion**: After developers, who next? Researchers? Writers? Everyone?

## Resources

- **Ollama**: https://ollama.ai
- **Llama Models**: https://huggingface.co/meta-llama
- **ChromaDB Docs**: https://docs.trychroma.com
- **YC Application**: https://apply.ycombinator.com

## Contact / Support

- **GitHub**: [your repo]
- **Email**: [your email]
- **Twitter**: [your handle]

---

**Last Updated**: [Date]
**Status**: MVP 90% complete, fixing RAG scoring
**Next Session**: Fix RAG, test quality, decide on local vs cloud priority
```

---

# Prompt for Next Claude Session
```
Hi! I'm building Hyperthymesia - a local-first AI search and Q&A tool for developers that I'm planning to apply to Y Combinator with.

**VISION**: Universal search for everything (files, emails, Slack, etc.) but starting with code search as developer wedge.

**CURRENT STATUS**: 
- Hybrid search (keyword + semantic) works perfectly
- RAG Q&A with Ollama is 90% done but has a scoring bug
- MVP almost ready for beta users

**THE PROBLEM I'M SOLVING NOW**:
The `ask` command (RAG Q&A) is finding node_modules documentation instead of user's actual code files. The `search` command works correctly and finds the right files.

**CONTEXT**:
- Search filters dependencies and finds: `config/db.js` ✅
- Ask doesn't filter and finds: `node_modules/...` docs ❌
- Both use same underlying data (ChromaDB + SQLite)

**WHAT I NEED HELP WITH**:
I added a dependency filter to `core/rag_retriever.py` in the `retrieve_context()` method but need to verify it will work correctly. The filter should:
1. Exclude paths containing: node_modules, vendor, .venv, site-packages, etc.
2. Boost remaining user code files 2x
3. Handle both semantic results (have 'metadata') and keyword results (have 'path' directly)

**KEY FILES**:
- `core/rag_retriever.py` - Needs the dependency filter fix
- `core/intelligent_searcher.py` - Query rewriting (works for search)
- `core/local_llm.py` - Ollama integration (working)
- `cli/commands.py` - Ask and search commands

**PROJECT INFO**:
- Tech: Python, SQLite FTS5, ChromaDB, Ollama (Llama 3.2)
- CLI tool, macOS development
- Database: `~/Library/Application Support/hyperthymesia/`

