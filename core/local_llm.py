"""
Local LLM integration for Q&A without cloud APIs.
Supports multiple backends: llama-cpp-python, MLX (Mac), Ollama.
Includes automatic model downloading and caching.
"""

import os
import platform
from pathlib import Path
from typing import Dict, List, Optional

from core.model_manager import ModelManager


class LocalLLM:
    """
    Local LLM wrapper supporting multiple backends.

    Backends (auto-detected):
    1. Ollama (easiest - if installed)
    2. MLX (Mac - Apple Silicon optimized)
    3. llama-cpp-python (cross-platform)
    """

    def __init__(self, model_name: str = "llama3.2:3b", auto_download: bool = True):
        """
        Initialize local LLM.

        Args:
            model_name: Model to use (ollama format: "llama3.2:3b")
                       Will be automatically converted to appropriate format for each backend
            auto_download: Whether to auto-download models if needed
        """
        self.ollama_model_name = model_name  # Keep original for Ollama
        # MLX uses HuggingFace model IDs - using a confirmed working model
        self.mlx_model_name = "mlx-community/llama2-7b-qnt4bit"
        self.backend = None
        self.model = None
        self.model_manager = ModelManager()
        self.auto_download = auto_download

        # Try to initialize
        self._initialize()

    def _initialize(self):
        """Try to initialize with available backend."""

        # Try Ollama first (easiest)
        if self._try_ollama():
            return

        # Try llama-cpp-python
        if self._try_llama_cpp():
            return

        # Try MLX on Mac (last resort - can be finicky)
        if platform.system() == "Darwin" and self._try_mlx():
            return

        print("⚠️  No local LLM backend available.")
        print("   Install one of:")
        print("   • Ollama: https://ollama.ai (easiest)")
        print("   • pip install llama-cpp-python")
        print("   • pip install mlx-lm (Mac only)")

    def _try_ollama(self) -> bool:
        """Try to use Ollama (simplest option)."""
        try:
            import requests

            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=200)
            if response.status_code == 200:
                self.backend = "ollama"
                print(f"✓ Using Ollama backend")
                return True
        except:
            pass

        return False

    def _try_mlx(self) -> bool:
        """Try to use MLX (Mac optimized)."""
        try:
            from mlx_lm import generate, load

            self.backend = "mlx"
            print(f"✓ Using MLX backend (Apple Silicon optimized)")
            print(f"  Model: {self.mlx_model_name}")
            print(f"  Note: First generation will load model (~30-60s)")

            # Model will be loaded on first use (lazy loading)
            return True
        except ImportError as e:
            print(f"  MLX not available: {e}")
            return False
        except Exception as e:
            print(f"  MLX initialization error: {e}")
            return False

    def _try_llama_cpp(self) -> bool:
        """Try to use llama-cpp-python with auto-download support."""
        try:
            from llama_cpp import Llama

            self.backend = "llama-cpp"
            print(f"✓ Using llama-cpp-python backend")

            # Try to find an existing model
            model_path = self._find_gguf_model()

            # If no model found and auto-download enabled, download one
            if not model_path and self.auto_download:
                print(f"  No local model found")
                print(f"\n  Would you like to download a model? (y/n) ", end="")
                try:
                    response = input().strip().lower()
                    if response == "y":
                        suggested = self.model_manager.suggest_model("llama-cpp")
                        print(f"\n  Downloading {suggested}...")
                        model_path = self.model_manager.download_model(
                            suggested, "llama-cpp"
                        )
                    else:
                        print(f"\n  Model download skipped")
                        self.model_manager.print_setup_instructions("llama-cpp")
                        return False
                except EOFError:
                    # Running non-interactively
                    print(f"\n  Attempting auto-download...")
                    suggested = self.model_manager.suggest_model("llama-cpp")
                    model_path = self.model_manager.download_model(
                        suggested, "llama-cpp"
                    )

            if model_path:
                print(f"  Loading model: {model_path}")
                try:
                    self.model = Llama(model_path=model_path, n_gpu_layers=-1)
                    print(f"  ✓ Model loaded (GPU enabled)")
                    return True
                except Exception as e:
                    print(f"  ⚠️  GPU loading failed: {e}")
                    try:
                        self.model = Llama(model_path=model_path)
                        print(f"  ✓ Model loaded (CPU mode)")
                        return True
                    except Exception as e2:
                        print(f"  ❌ Failed to load model: {e2}")
                        return False
            else:
                print(f"  ❌ No model available")
                self.model_manager.print_setup_instructions("llama-cpp")
                return False

        except ImportError as e:
            print(f"  ❌ llama-cpp-python not installed")
            print(f"     Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            print(f"  ❌ llama-cpp error: {e}")
            return False

    def _find_gguf_model(self) -> Optional[str]:
        """Find a GGUF model file in standard location."""
        models_dir = Path.home() / ".hyperthymesia" / "models"

        if models_dir.exists():
            for model_file in models_dir.glob("*.gguf"):
                return str(model_file)

        return None

    def is_available(self) -> bool:
        """Check if local LLM is available."""
        return self.backend is not None

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text from a prompt using the available backend.

        Generic method that works with any available backend (Ollama, MLX, llama-cpp).

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        if not self.is_available():
            return (
                "❌ No local LLM available. Please install Ollama or llama-cpp-python."
            )

        # Route to appropriate backend
        if self.backend == "ollama":
            return self._generate_ollama(prompt, max_tokens)
        elif self.backend == "mlx":
            return self._generate_mlx(prompt, max_tokens)
        elif self.backend == "llama-cpp":
            return self._generate_llama_cpp(prompt, max_tokens)

        return "❌ Unknown backend"

    def answer_question(
        self, question: str, context: str, max_tokens: int = 500
    ) -> str:
        """
        Answer a question using retrieved context.

        Args:
            question: User's question
            context: Retrieved context from documents
            max_tokens: Maximum tokens to generate

        Returns:
            Generated answer
        """
        if not self.is_available():
            return (
                "❌ No local LLM available. Please install Ollama or llama-cpp-python."
            )

        # Build prompt
        prompt = self._build_prompt(question, context)

        # Use generic generate method (works with all backends)
        return self.generate(prompt, max_tokens)

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt for developer Q&A.

        Optimized for answering code-related questions with focus on implementation details.
        """
        prompt = f"""You are an expert code assistant helping a developer understand their codebase.

Retrieved code context:
{context}

Developer's question: {question}

Guidelines:
• Answer based ONLY on the provided code context
• Be specific about class names, function names, and file locations
• Explain the implementation, not just what it does
• If code examples are shown, reference them directly
• If the context doesn't have the answer, clearly state that
• Keep answers concise but complete (2-4 sentences)

Answer:"""

        return prompt

    def _generate_ollama(self, prompt: str, max_tokens: int) -> str:
        """Generate using Ollama API."""
        import requests

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.7},
                },
                timeout=300,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"❌ Ollama error: {response.status_code}"

        except Exception as e:
            return f"❌ Error: {e}"

    def _generate_mlx(self, prompt: str, max_tokens: int) -> str:
        """Generate using MLX (Mac optimized)."""
        try:
            from mlx_lm import generate, load

            # Lazy load model
            if self.model is None:
                print(f"\n⏳ Loading MLX model: {self.mlx_model_name}")
                print(f"   This may take 30-60 seconds on first use...\n")
                try:
                    self.model, self.tokenizer = load(self.mlx_model_name)
                    print(f"✓ Model loaded successfully\n")
                except Exception as e:
                    return f"❌ Failed to load model: {e}\n\nMake sure the model exists. Try: mlx_lm.available_models()"

            try:
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return response.strip()
            except Exception as e:
                return f"❌ MLX generation error: {e}"

        except Exception as e:
            return f"❌ MLX error: {e}"

    def _generate_llama_cpp(self, prompt: str, max_tokens: int) -> str:
        """Generate using llama-cpp-python (CPU/GPU compatible)."""
        try:
            from llama_cpp import Llama

            # Model should have been loaded in _try_llama_cpp
            if self.model is None:
                return """❌ Model not loaded.

To use llama-cpp-python:
1. Download a GGUF model:
   https://huggingface.co/models?search=gguf

2. Create directory: mkdir -p ~/.hyperthymesia/models/

3. Download a model (e.g., Mistral 7B):
   wget -O ~/.hyperthymesia/models/model.gguf \\
     https://huggingface.co/.../model.gguf

4. Run agent again - it will auto-detect

Recommended models:
- mistral-7b-instruct (faster, good quality)
- neural-chat-7b (optimized for chat)
- openchat-3.5 (good balance)

Or use Ollama instead (easier): https://ollama.ai"""

            try:
                response = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    echo=False,
                    top_p=0.95,
                    top_k=40,
                )

                return response["choices"][0]["text"].strip()
            except Exception as e:
                return f"❌ llama-cpp generation error: {e}"

        except Exception as e:
            return f"❌ llama-cpp error: {e}"

    def get_backend_info(self) -> Dict:
        """Get information about the active backend."""
        info = {
            "backend": self.backend,
            "model": self.model_name,
            "available": self.is_available(),
        }

        # Add backend-specific info
        if self.backend == "ollama":
            info["status"] = "Running (API endpoint available)"
            info["instructions"] = "ollama serve"
        elif self.backend == "mlx":
            info["status"] = "Apple Silicon optimized"
            info["instructions"] = "pip install mlx-lm"
        elif self.backend == "llama-cpp":
            info["status"] = "CPU/GPU compatible"
            info["instructions"] = "pip install llama-cpp-python"

        return info

    def get_available_backends(self) -> Dict[str, bool]:
        """Check which backends are available."""
        backends = {
            "ollama": False,
            "mlx": False,
            "llama-cpp": False,
        }

        # Check ollama
        try:
            import requests

            requests.get("http://localhost:11434/api/tags", timeout=1)
            backends["ollama"] = True
        except:
            pass

        # Check mlx
        try:
            import mlx_lm

            if platform.system() == "Darwin":
                backends["mlx"] = True
        except:
            pass

        # Check llama-cpp
        try:
            import llama_cpp

            backends["llama-cpp"] = True
        except:
            pass

        return backends


# For BYOK (Bring Your Own Key) - Cloud fallback
class CloudLLM:
    """Cloud LLM fallback using user's API keys."""

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize cloud LLM.

        Args:
            provider: 'openai', 'anthropic', or 'gemini'
            api_key: User's API key
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        if not self.api_key:
            raise ValueError(f"No API key found for {provider}")

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return self.api_key is not None

    def answer_question(
        self, question: str, context: str, max_tokens: int = 500
    ) -> str:
        """Answer using cloud API."""

        if self.provider == "openai":
            return self._call_openai(question, context, max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(question, context, max_tokens)
        elif self.provider == "gemini":
            return self._call_gemini(question, context, max_tokens)

        return "❌ Unsupported provider"

    def _call_openai(self, question: str, context: str, max_tokens: int) -> str:
        """Call OpenAI API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer questions based on provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ OpenAI error: {e}"

    def _call_anthropic(self, question: str, context: str, max_tokens: int) -> str:
        """Call Anthropic API."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context:",
                    }
                ],
            )

            return message.content[0].text.strip()

        except Exception as e:
            return f"❌ Anthropic error: {e}"

    def _call_gemini(self, question: str, context: str, max_tokens: int) -> str:
        """Call Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

            response = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": 0.7},
            )

            return response.text.strip()

        except Exception as e:
            return f"❌ Gemini error: {e}"
