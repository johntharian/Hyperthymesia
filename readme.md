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
✓ **Multiple LLM Backends** - Auto-detects Ollama, llama-cpp, or MLX on Apple Silicon
✓ **Zero Configuration** - Auto-detects and selects backend/model—just install and run
✓ **Optimized Performance** - 3-4x faster inference with quantized models
✓ **Intelligent Query Routing** - Automatically chooses simple search or advanced analysis
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

### Setup LLM Backend (Automatic)

Hyperthymesia **automatically detects and selects** an LLM backend—no manual configuration needed!

**Supported backends** (auto-detected in order):
1. **Ollama** (easiest, recommended)
2. **llama-cpp-python** (cross-platform)

#### Option 1: Use Ollama (Recommended)

Visit [ollama.ai](https://ollama.ai) and download Ollama for your operating system:
- macOS: Download the `.dmg` file and install
- Linux: Run the installation script
- Windows: Download the Windows installer (or use WSL)

Then run Ollama in the background:
```bash
ollama serve
```

Leave this running. When you use Hyperthymesia, it will:
- Auto-detect Ollama is running
- Show you available models
- Let you choose which model to use (or auto-select)
- Work with any Ollama model (mistral, neural-chat, openhermes, etc.)

**Recommended models** (quantized for speed):
```bash
ollama pull mistral:latest           # Fast & high quality
ollama pull neural-chat:latest       # Optimized for chat
ollama pull openhermes:latest        # Good balance
```

#### Option 2: Use llama-cpp-python (No Extra Installation)

If you have GGUF model files locally at `~/.hyperthymesia/models/`, Hyperthymesia will auto-detect and use them:
- No server needed
- Runs directly in Python
- GPU acceleration available (CUDA, Metal)
- Auto-downloads recommended models on first run

### First Use

**Step 1: Index a codebase** (one-time setup)

```bash
hyperthymesia index add /path/to/your/project
```

This analyzes your code and creates search indices. It runs once per project.

**Note**: If the project is already indexed, it won't be re-indexed—just use the cached index for fast queries.

**Step 2: Search for code**

```bash
# Simple keyword search
hyperthymesia search "authentication"

# Semantic search (finds related code even with different names)
hyperthymesia search "user login flow"
```

**Step 3: Ask complex questions**

```bash
# Simple questions use fast keyword search
hyperthymesia ask "Where is the authentication code?"

# Complex questions automatically use the AI agent with code analysis
hyperthymesia ask "How does the authentication flow work across the codebase?"

# Show reasoning with verbose flag
hyperthymesia ask "Explain the login mechanism" --verbose
```

The system **automatically decides** whether to use simple search or advanced agent analysis based on question complexity.

## Features & Usage

### Simple Search

Find code with keyword and semantic search:
```bash
hyperthymesia search "database connection"
hyperthymesia search "error handling"
hyperthymesia search "cache mechanism"
```

### Agent Queries

Ask complex questions and get AI-powered analysis. The system automatically routes complex questions to agent mode:

```bash
# Understand implementation
hyperthymesia ask "How does authentication work?"

# Find patterns
hyperthymesia ask "Where is error handling implemented?"

# Explain mechanisms
hyperthymesia ask "Explain the retry mechanism"

# Explore architecture
hyperthymesia ask "What is the overall architecture?"

# See the reasoning process (debug/transparency)
hyperthymesia ask "How does the payment flow work?" --verbose
```

**With `--verbose` flag**, you'll see:
- The LLM's reasoning about your question
- Search queries generated by the agent
- Code analysis steps
- Final synthesis of the answer

This is useful for understanding how the agent approaches complex questions.

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

Hyperthymesia automatically detects and uses the best available backend:

| Backend | Install | Speed | Quality | Best For |
|---------|---------|-------|---------|----------|
| **Ollama** | Download & run | ⚡⚡⚡ | ⭐⭐⭐⭐ | Most users (easiest) |
| **llama-cpp** | pip install (Python) | ⚡⚡⚡ | ⭐⭐⭐⭐ | Quantized models (q4_K_M) |
| **MLX** | pip install (macOS) | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Apple Silicon optimization |

**Performance with Quantized Models (q4_K_M)**:
- Token generation: 20-40 ms/token (GPU), 40-60 ms/token (CPU)
- Memory usage: ~7GB for 7B model, ~14GB for 13B model
- Inference speedup: 3-4x faster than full-precision models

**Why auto-detection matters**:
- No configuration needed—just install and run
- Falls back gracefully (Ollama → llama-cpp → MLX)
- Uses the fastest available backend automatically
- Single model selection prompt (if needed)

## How It Works

### Backend Auto-Detection Flow

When you run Hyperthymesia, it automatically:

```
1. Check if Ollama is running (localhost:11434)
   └─ If yes: Auto-detect available models → Let user choose (or auto-select)
   └─ If no: Continue to next backend

2. Check if llama-cpp-python is installed
   └─ If yes: Look for GGUF models in ~/.hyperthymesia/models/
   └─ If found: Load with GPU acceleration (fallback to CPU)
   └─ If not: Offer to download recommended model

3. Check if MLX is available (macOS only)
   └─ If yes: Use for Apple Silicon optimization

4. If no backend available
   └─ Fallback to keyword search only (no AI answers)
```

**Zero configuration needed** — just install and use!

### Intelligent Query Routing

Hyperthymesia automatically chooses the right tool for each query:

```
User Question
     ↓
Is it a simple, direct question?
     ├─ Yes → Fast keyword search (instant)
     └─ No → Complex analysis with AI agent (uses reasoning + multi-tool orchestration)

Complex analysis flow:
1. Use LLM to generate reasoning about the question
2. Create a search plan based on reasoning
3. Execute searches (hybrid keyword + semantic)
4. Analyze results with code structure extraction
5. Synthesize comprehensive answer
```

### Optimized LLM Performance

The system uses several techniques to improve llama-cpp performance:

- **Concise Prompts**: Tailored to work well with smaller models
- **Lower Temperature (0.3)**: Ensures factual consistency over creativity
- **Repetition Penalties**: Prevents filler text and redundant answers
- **Chat Completion API**: Structured format reduces rambling by ~80%
- **Quantized Models**: q4_K_M quantization provides 3-4x speedup
- **GPU Acceleration**: Offloads computation for faster inference

Result: **High-quality answers in 2-5 seconds** instead of repetitive output.

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
