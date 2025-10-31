# Hyperthymesia Installation Guide

## Quick Install (macOS)

The easiest way to install Hyperthymesia with all dependencies:

```bash
bash install.sh
```

That's it! The installer will:
1. ✓ Install Homebrew (if needed)
2. ✓ Install Ollama
3. ✓ Pull the `llama3.2:3b` model
4. ✓ Install Hyperthymesia and Python dependencies
5. ✓ Verify everything is working

**Installation takes ~5-15 minutes** (mostly downloading the LLM model)

## What Gets Installed

### Ollama
- LLM service that runs in the background
- Model: `llama3.2:3b` (~2GB)
- Runs on: http://localhost:11434

### Hyperthymesia
- Local search and code assistant
- Python package with all dependencies
- CLI tools for indexing and querying code

## After Installation

### 1. Index Your Code

```bash
# Index a single project
hyperthymesia index add /path/to/project

# Index multiple projects
hyperthymesia index add /path/to/project1
hyperthymesia index add /path/to/project2

# List indexed projects
hyperthymesia index list
```

### 2. Search Your Code

```bash
# Simple search
hyperthymesia search "authentication"

# Agent-powered intelligent search
hyperthymesia agent "how does authentication work?"

# Verbose mode (see reasoning)
hyperthymesia agent "your question" --verbose
```

## Troubleshooting

### Ollama won't start
```bash
# Check if it's running
brew services list | grep ollama

# Restart Ollama
brew services restart ollama

# Check logs
log stream --predicate 'eventMessage contains "ollama"'
```

### Model already downloaded?
```bash
# See what models you have
ollama list

# Remove a model
ollama rm llama3.2:3b

# Pull a different model
ollama pull mistral
```

### Python import errors
```bash
# Reinstall dependencies
cd hyperthymesia_cli
pip install -e .

# Verify imports
python3 -c "from core.agent import HyperthymesiaAgent; print('OK')"
```

### Permission denied on install.sh
```bash
chmod +x install.sh
bash install.sh
```

## Manual Installation

If the automated installer doesn't work, follow these steps:

### Step 1: Install Ollama

```bash
brew install ollama
brew services start ollama
```

### Step 2: Pull Model

```bash
ollama pull llama3.2:3b
```

### Step 3: Install Hyperthymesia

```bash
# Clone the repository
git clone https://github.com/yourusername/hyperthymesia.git
cd hyperthymesia/hyperthymesia_cli

# Install Python dependencies
pip install -e .
```

### Step 4: Verify

```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# Test Hyperthymesia
hyperthymesia --help
```

## Alternative Backends

If Ollama doesn't work for you, try:

### llama-cpp-python
```bash
pip install llama-cpp-python

# Models auto-download on first use
hyperthymesia agent "your question"
```

### MLX (Apple Silicon only)
```bash
pip install mlx-lm

# Will auto-download and use MLX models
hyperthymesia agent "your question"
```

## System Requirements

### Minimum
- macOS 10.14 or later
- 8GB RAM
- 5GB free disk space (for models)
- Python 3.8+

### Recommended
- macOS 12 or later
- 16GB RAM
- 10GB free disk space
- Apple Silicon (M1/M2/M3) for best MLX performance

## Getting Help

- Check `/path/to/hyperthymesia/SETUP.txt` for detailed setup info
- Read `AGENT_CLI_USAGE.md` for command examples
- See `AGENT_ARCHITECTURE.md` for technical details

## Uninstalling

```bash
# Remove Hyperthymesia
pip uninstall hyperthymesia

# Remove Ollama
brew uninstall ollama

# Remove Ollama data (optional)
rm -rf ~/.ollama
```

## What's Next?

1. Index your codebase: `hyperthymesia index add /path/to/code`
2. Try a search: `hyperthymesia agent "how does X work?"`
3. Use verbose mode: `hyperthymesia agent "..." --verbose`
4. Explore the documentation

Happy searching! 🔍
