#!/bin/bash

################################################################################
# Hyperthymesia Installer - macOS
#
# This script automates the installation of Hyperthymesia with Ollama backend
# on macOS. It handles all dependencies and setup without user input.
#
# Usage: bash install.sh
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Check if running on macOS
check_macos() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "This installer is for macOS only"
        exit 1
    fi
    log_success "Running on macOS"
}

# Check if Homebrew is installed, install if needed
check_homebrew() {
    if ! command -v brew &> /dev/null; then
        log_info "Homebrew not found, installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for M1/M2 Macs
        if [[ $(uname -m) == 'arm64' ]]; then
            export PATH="/opt/homebrew/bin:$PATH"
        fi
    fi
    log_success "Homebrew is ready"
}

# Install Ollama
install_ollama() {
    if command -v ollama &> /dev/null; then
        log_warning "Ollama is already installed"
        return
    fi

    log_info "Installing Ollama via Homebrew..."
    brew install ollama
    log_success "Ollama installed"
}

# Start Ollama service
start_ollama() {
    log_info "Starting Ollama service..."

    # Start ollama in background if not already running
    if ! pgrep -x "ollama" > /dev/null; then
        # Start as a launchd service
        brew services start ollama

        # Wait for Ollama to be ready
        log_info "Waiting for Ollama to start..."
        sleep 3

        # Check if it's running
        if pgrep -x "ollama" > /dev/null; then
            log_success "Ollama service started"
        else
            log_error "Failed to start Ollama service"
            log_info "Try manually: brew services start ollama"
            exit 1
        fi
    else
        log_success "Ollama is already running"
    fi
}

# Pull the model
pull_model() {
    log_info "Pulling llama3.2:3b model (this may take a few minutes)..."

    # Check if model already exists
    if ollama list | grep -q "llama3.2:3b"; then
        log_warning "Model llama3.2:3b is already downloaded"
        return
    fi

    # Pull the model
    ollama pull llama3.2:3b
    log_success "Model llama3.2:3b pulled successfully"
}

# Check Python installation
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        log_info "Install Python 3 via Homebrew: brew install python3"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_success "Python 3 found: $PYTHON_VERSION"
}

# Clone or locate Hyperthymesia repo
setup_hyperthymesia_repo() {
    if [ -d "hyperthymesia_cli" ]; then
        log_info "Hyperthymesia repository found in current directory"
        cd hyperthymesia_cli
    elif [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        log_info "Already in Hyperthymesia directory"
    else
        log_info "Cloning Hyperthymesia repository..."

        if ! git clone https://github.com/yourusername/hyperthymesia.git hyperthymesia_cli 2>/dev/null; then
            log_warning "Could not clone from GitHub. Make sure you're in the repo directory or update the git URL"
            if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
                log_error "Hyperthymesia not found and could not be cloned"
                exit 1
            fi
        else
            cd hyperthymesia_cli
        fi
    fi
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    # Upgrade pip
    python3 -m pip install --upgrade pip

    # Install Hyperthymesia in development mode
    python3 -m pip install -e .

    log_success "Python dependencies installed"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Check if hyperthymesia command is available
    if command -v hyperthymesia &> /dev/null; then
        log_success "hyperthymesia command is available"
    else
        log_warning "hyperthymesia command not in PATH"
        log_info "Try: python3 -m hyperthymesia --help"
    fi

    # Test Python imports
    if python3 -c "from core.local_llm import LocalLLM; from core.agent import HyperthymesiaAgent" 2>/dev/null; then
        log_success "Python modules imported successfully"
    else
        log_error "Failed to import required modules"
        exit 1
    fi

    # Test Ollama connection
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_success "Ollama is running and accessible"
    else
        log_warning "Could not verify Ollama connection"
        log_info "Ensure Ollama is running: brew services start ollama"
    fi
}

# Show success message and next steps
show_success() {
    log_step "Installation Complete! 🎉"

    echo -e "Hyperthymesia is now installed and ready to use!\n"

    echo -e "${YELLOW}Quick Start:${NC}"
    echo -e "  1. Index your code:"
    echo -e "     hyperthymesia index add /path/to/your/code\n"

    echo -e "  2. Ask a question:"
    echo -e "     hyperthymesia agent \"how does authentication work?\"\n"

    echo -e "  3. See reasoning (verbose mode):"
    echo -e "     hyperthymesia agent \"your question\" --verbose\n"

    echo -e "${YELLOW}Useful Commands:${NC}"
    echo -e "  hyperthymesia search \"keyword\"       - Search indexed code"
    echo -e "  hyperthymesia agent \"question\"       - Ask agent a question"
    echo -e "  hyperthymesia index list              - List indexed projects"
    echo -e "  hyperthymesia --help                  - Show all commands\n"

    echo -e "${YELLOW}Ollama Management:${NC}"
    echo -e "  brew services start ollama            - Start Ollama"
    echo -e "  brew services stop ollama             - Stop Ollama"
    echo -e "  ollama list                           - List available models"
    echo -e "  ollama pull llama3.2:3b               - Pull a specific model\n"

    echo -e "${YELLOW}Documentation:${NC}"
    echo -e "  SETUP.txt                 - Detailed setup instructions"
    echo -e "  AGENT_CLI_USAGE.md        - Agent command usage guide"
    echo -e "  AGENT_ARCHITECTURE.md     - Technical architecture\n"

    echo -e "${GREEN}Happy searching! 🔍${NC}\n"
}

# Handle errors
trap 'log_error "Installation failed at line $LINENO"; exit 1' ERR

################################################################################
# Main Installation Flow
################################################################################

log_step "Hyperthymesia Installer"

check_macos
check_homebrew
check_python
install_ollama
start_ollama
pull_model
setup_hyperthymesia_repo
install_dependencies
verify_installation
show_success

log_step "All Done!"
log_success "Hyperthymesia is ready to use"
