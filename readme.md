# Hyperthymesia - Local AI-Powered Code Search & Understanding

Hyperthymesia is a **local-first, privacy-focused AI assistant** that helps developers understand, search, and navigate their codebases using intelligent AI. Everything runs on your machine—your code never leaves your computer.

## What is Hyperthymesia?

Hyperthymesia combines hybrid search (keyword + semantic), an intelligent AI agent, and local LLMs to help you:
- **Search your code** with both keyword and semantic understanding
- **Ask complex questions** like "How does authentication work?" and get detailed, multi-step analysis
- **Understand patterns** in your codebase without uploading to the cloud
- **Onboard faster** to new projects by exploring architecture and code flows

**Key advantage**: 100% local processing. Your code never leaves your machine.

## Key Features

✓ **Hybrid Search** - Keyword + semantic search combined for better results
✓ **Intelligent Agent** - Multi-tool AI orchestration for complex code questions
✓ **100% Private** - No cloud, no subscriptions, no data leaving your machine
✓ **Works Offline** - No internet required after initial setup
✓ **Multiple LLM Backends** - Use Ollama, llama-cpp, or MLX on Apple Silicon
✓ **Fast** - Local processing means instant responses

## Quick Start

### Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 5GB for LLM models
- **OS**: macOS (Intel/Apple Silicon), Linux, or Windows (WSL)

### Installation Steps

**Step 1: Clone the repository**
```bash
git clone https://github.com/johntharian/hyperthymesia.git
cd hyperthymesia_cli
```

**Step 2: Install Hyperthymesia**
```bash
pip install -e .
```

This installs Hyperthymesia and all its dependencies in development mode.

### Setup Ollama & LLaMA Model

Hyperthymesia needs a local LLM to run. We recommend **Ollama** as it's the easiest to set up.

**Step 1: Install Ollama**

Visit [ollama.ai](https://ollama.ai) and download Ollama for your operating system:
- macOS: Download the `.dmg` file and install
- Linux: Run the installation script
- Windows: Download the Windows installer (or use WSL)

**Step 2: Pull the LLaMA model**

Open a terminal and run:
```bash
ollama pull llama3.2:3b
```

**Step 3: Run Ollama in the background**

```bash
ollama serve
```

Leave this running. Ollama will serve the model on `localhost:11434`.

### First Use

**Step 1: Index a codebase**

```bash
hyperthymesia index add /path/to/your/project
```

This analyzes your code and creates search indices. It runs once per project.

**Step 2: Try a simple search**

```bash
hyperthymesia search "authentication"
```

Returns files that match "authentication" using both keyword and semantic search.

**Step 3: Ask the AI agent a question**

```bash
hyperthymesia agent "How does authentication work in this codebase?"
```

The agent will:
1. Search for relevant code
2. Analyze the structure
3. Generate a detailed explanation with code references

## Features & Usage

### Simple Search

Find code with keyword and semantic search:
```bash
hyperthymesia search "database connection"
hyperthymesia search "error handling"
hyperthymesia search "cache mechanism"
```

### Agent Queries

Ask complex questions and get AI-powered analysis:
```bash
# Understand implementation
hyperthymesia agent "How does authentication work?"

# Find patterns
hyperthymesia agent "Where is error handling implemented?"

# Explain mechanisms
hyperthymesia agent "Explain the retry mechanism"

# Explore architecture
hyperthymesia agent "What is the overall architecture?"
```

### Configuration

View or modify Hyperthymesia settings:
```bash
# Show current configuration
hyperthymesia config show

# Set LLM backend
hyperthymesia config set llm-backend ollama

# List indexed projects
hyperthymesia index list

# Remove a project from index
hyperthymesia index remove /path/to/project
```

## System Requirements

### Supported Operating Systems

- **macOS**: 10.14+ (Intel and Apple Silicon)
- **Linux**: Most distributions (Ubuntu, Debian, Fedora, etc.)
- **Windows**: Windows Subsystem for Linux (WSL) recommended

### Hardware

- **Minimum**: 8GB RAM, 5GB disk space
- **Recommended**: 16GB RAM, SSD for faster indexing
- **GPU**: Optional, but can speed up LLM inference

### LLM Backend Options

Choose one of these:

| Backend | Install | Pros | Cons |
|---------|---------|------|------|
| **Ollama** (recommended) | Easy (download & run) | User-friendly, fast | macOS/Linux/Windows WSL |
| **llama-cpp** | `pip install llama-cpp-python` | Cross-platform, no installation | More configuration |
| **MLX** | `pip install mlx-lm` | Apple Silicon optimized | macOS only |

## Troubleshooting

### Installation Issues

#### Permission Errors
If you encounter permission errors during installation:

1. Use a virtual environment (recommended):
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate it
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # .\venv\Scripts\activate
   
   # Then install
   pip install -e .
   
### Ollama not connecting

Make sure Ollama is running:
```bash
# Check if Ollama is serving
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### Model not found

Pull the model:
```bash
ollama pull llama2
```

### Indexing slow

This is normal for large codebases. Use `--verbose` to see progress:
```bash
hyperthymesia index add /path/to/project --verbose
```

### Out of memory

If you run out of memory, use a smaller model:
```bash
ollama pull llama2:3b-chat-q4_0
```

## Documentation

For more detailed information, see:
- [PRODUCT_DESCRIPTION.md](PRODUCT_DESCRIPTION.md) - Full feature overview and roadmap
- [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md) - Technical architecture details

## Privacy & Security

- **100% Local Processing** - Your code never leaves your machine
- **No Cloud Services** - No API calls, no data uploads
- **No Tracking** - No telemetry or analytics
- **Open Source** - Code is transparent and auditable

## License

Open source and free to use.

## Contributing

Contributions are welcome! Open an issue or submit a pull request.
