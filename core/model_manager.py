"""
Model manager for automatic LLM model downloading and caching.
Supports downloading GGUF models for llama-cpp-python and MLX.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple


class ModelManager:
    """Manages downloading and caching of LLM models."""

    # Popular small models for different purposes
    RECOMMENDED_MODELS = {
        "llama-cpp": {
            "mistral-7b-instruct": {
                "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/Mistral-7B-Instruct-v0.2.Q4_K_M.gguf",
                "size": "4.37GB",
                "description": "Fast, good quality, recommended for CPU",
            },
            "neural-chat-7b": {
                "url": "https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF/resolve/main/neural-chat-7B-v3-3.Q4_K_M.gguf",
                "size": "4.29GB",
                "description": "Optimized for chat",
            },
            "openchat-3.5": {
                "url": "https://huggingface.co/TheBloke/OpenChat-3.5-GGUF/resolve/main/openchat-3.5.Q4_K_M.gguf",
                "size": "4.16GB",
                "description": "Good balance of speed and quality",
            },
            "llama-3.2-3b.Q4_K_M.gguf": {
                "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ3_M.gguf",
                "size": "4.16GB",
                "description": "Good balance of speed and quality",
            },
        },
        "mlx": {
            "mistral-7b": {
                "description": "Auto-downloads via mlx-lm",
                "command": "mlx_lm.generate",
            },
            "llama-2-7b": {
                "description": "Auto-downloads via mlx-lm",
                "command": "mlx_lm.generate",
            },
        },
    }

    def __init__(self):
        """Initialize model manager."""
        self.models_dir = Path.home() / ".hyperthymesia" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.models_dir / "models.json"

    def get_available_models(self, backend: str) -> Dict:
        """Get available models for a backend."""
        return self.RECOMMENDED_MODELS.get(backend, {})

    def list_local_models(self) -> Dict[str, str]:
        """List all locally cached models."""
        models = {}
        for model_file in self.models_dir.glob("*.gguf"):
            size = model_file.stat().st_size / (1024**3)  # Convert to GB
            models[model_file.name] = f"{size:.2f}GB"
        return models

    def has_model(self, model_name: str) -> bool:
        """Check if model is already downloaded."""
        model_path = self.models_dir / f"{model_name}.gguf"
        return model_path.exists()

    def download_model(
        self, model_key: str, backend: str = "llama-cpp", progress_callback=None
    ) -> Optional[str]:
        """
        Download a model.

        Args:
            model_key: Model identifier (e.g., 'mistral-7b-instruct')
            backend: Backend type ('llama-cpp' or 'mlx')
            progress_callback: Optional callback for download progress

        Returns:
            Path to downloaded model or None if failed
        """
        models = self.get_available_models(backend)

        if model_key not in models:
            print(f"❌ Model '{model_key}' not found for {backend}")
            print(f"   Available models: {', '.join(models.keys())}")
            return None

        model_info = models[model_key]

        # For MLX, models are auto-downloaded by mlx-lm
        if backend == "mlx":
            print(f"✓ MLX will auto-download '{model_key}' on first use")
            return None

        # For llama-cpp, download GGUF file
        if backend == "llama-cpp":
            url = model_info.get("url")
            if not url:
                print(f"❌ No download URL for {model_key}")
                return None

            model_path = self.models_dir / f"{model_key}.gguf"

            if model_path.exists():
                print(f"✓ Model already downloaded: {model_path}")
                return str(model_path)

            print(f"\n📥 Downloading {model_key} ({model_info.get('size', 'unknown')})")
            print(f"   This may take a few minutes...\n")

            try:
                return self._download_file(url, model_path, progress_callback)
            except Exception as e:
                print(f"❌ Download failed: {e}")
                return None

        return None

    def _download_file(
        self, url: str, destination: Path, progress_callback=None
    ) -> Optional[str]:
        """
        Download a file with progress reporting.

        Args:
            url: URL to download from
            destination: Path to save file
            progress_callback: Optional callback(bytes_downloaded, total_bytes)

        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Open URL
            response = urllib.request.urlopen(url, timeout=30)
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress
            chunk_size = 8192  # 8KB chunks
            downloaded = 0

            with open(destination, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback:
                        progress_callback(downloaded, total_size)

                    # Print progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total_size)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        print(f"\r  [{bar}] {percent:.1f}%", end="", flush=True)

            print()  # New line after progress bar
            print(f"✓ Downloaded to: {destination}")
            return str(destination)

        except urllib.error.URLError as e:
            print(f"❌ URL error: {e}")
            return None
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    def get_model_path(self, model_key: str) -> Optional[str]:
        """Get path to a downloaded model."""
        model_path = self.models_dir / f"{model_key}.gguf"
        if model_path.exists():
            return str(model_path)
        return None

    def suggest_model(self, backend: str) -> str:
        """Suggest a model for the given backend."""
        if backend == "llama-cpp":
            return "llama-3.2-3b.Q4_K_M.gguf"
        elif backend == "mlx":
            return "mistral-7b"
        return ""

    def print_setup_instructions(self, backend: str):
        """Print setup instructions for a backend."""
        if backend == "llama-cpp":
            print("\n" + "=" * 60)
            print("📦 LLM Setup for llama-cpp-python")
            print("=" * 60)
            print("\n1. Install llama-cpp-python:")
            print("   pip install llama-cpp-python\n")

            print("2. Download a model:")
            print("   Option A - Automatic (during first run):")
            print("      hyperthymesia agent 'your question'")
            print("      (will prompt to download)\n")
            print("   Option B - Manual:")
            models = self.get_available_models("llama-cpp")
            for key, info in list(models.items())[:3]:
                print(f"      - {key}: {info['description']} ({info['size']})")
            print("\n3. Models are cached in: ~/.hyperthymesia/models/\n")

        elif backend == "mlx":
            print("\n" + "=" * 60)
            print("🍎 LLM Setup for MLX (Apple Silicon)")
            print("=" * 60)
            print("\n1. Install mlx-lm:")
            print("   pip install mlx-lm\n")

            print("2. Models auto-download on first use")
            print("   Default: mistral-7b\n")

            print("3. Supported models:")
            for model_key in ["mistral-7b", "llama-2-7b"]:
                print(f"   - {model_key}")
            print()

        elif backend == "ollama":
            print("\n" + "=" * 60)
            print("🦙 LLM Setup for Ollama")
            print("=" * 60)
            print("\n1. Install Ollama: https://ollama.ai\n")

            print("2. Pull a model:")
            print("   ollama pull llama2")
            print("   ollama pull mistral")
            print("   ollama pull neural-chat\n")

            print("3. Start Ollama:")
            print("   ollama serve\n")

            print("4. Then run agent:")
            print("   hyperthymesia agent 'your question'\n")

        print("=" * 60 + "\n")
