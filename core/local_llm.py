"""
Local LLM integration for Q&A without cloud APIs.
Supports multiple backends: llama-cpp-python, MLX (Mac), Ollama.
"""
import os
import platform
from typing import Optional, List, Dict
from pathlib import Path


class LocalLLM:
    """
    Local LLM wrapper supporting multiple backends.
    
    Backends (auto-detected):
    1. Ollama (easiest - if installed)
    2. MLX (Mac - Apple Silicon optimized)
    3. llama-cpp-python (cross-platform)
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize local LLM.
        
        Args:
            model_name: Model to use (ollama format: "llama3.2:3b")
        """
        self.model_name = model_name
        self.backend = None
        self.model = None
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self):
        """Try to initialize with available backend."""
        
        # Try Ollama first (easiest)
        if self._try_ollama():
            return
        
        # Try MLX on Mac
        if platform.system() == 'Darwin' and self._try_mlx():
            return
        
        # Try llama-cpp-python
        if self._try_llama_cpp():
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
            response = requests.get('http://localhost:11434/api/tags', timeout=200)
            if response.status_code == 200:
                self.backend = 'ollama'
                print(f"✓ Using Ollama backend")
                return True
        except:
            pass
        
        return False
    
    def _try_mlx(self) -> bool:
        """Try to use MLX (Mac optimized)."""
        try:
            from mlx_lm import load, generate
            
            self.backend = 'mlx'
            print(f"✓ Using MLX backend (Apple Silicon optimized)")
            
            # Model will be loaded on first use (lazy loading)
            return True
        except ImportError:
            return False
    
    def _try_llama_cpp(self) -> bool:
        """Try to use llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            self.backend = 'llama-cpp'
            print(f"✓ Using llama-cpp-python backend")
            
            # Model will be loaded on first use
            return True
        except ImportError:
            return False
    
    def is_available(self) -> bool:
        """Check if local LLM is available."""
        return self.backend is not None
    
    def answer_question(self, question: str, context: str, 
                       max_tokens: int = 500) -> str:
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
            return "❌ No local LLM available. Please install Ollama or llama-cpp-python."
        
        # Build prompt
        prompt = self._build_prompt(question, context)
        
        # Generate with appropriate backend
        if self.backend == 'ollama':
            return self._generate_ollama(prompt, max_tokens)
        elif self.backend == 'mlx':
            return self._generate_mlx(prompt, max_tokens)
        elif self.backend == 'llama-cpp':
            return self._generate_llama_cpp(prompt, max_tokens)
        
        return "❌ Unknown backend"
    
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
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': max_tokens,
                        'temperature': 0.7
                    }
                },
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"❌ Ollama error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Error: {e}"
    
    def _generate_mlx(self, prompt: str, max_tokens: int) -> str:
        """Generate using MLX."""
        try:
            from mlx_lm import load, generate
            
            # Lazy load model
            if self.model is None:
                print("Loading model (first time, ~30s)...")
                self.model, self.tokenizer = load(self.model_name)
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=0.7
            )
            
            return response.strip()
            
        except Exception as e:
            return f"❌ MLX error: {e}"
    
    def _generate_llama_cpp(self, prompt: str, max_tokens: int) -> str:
        """Generate using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            # Lazy load model
            if self.model is None:
                # User needs to download model manually
                # We'll give instructions
                return """❌ Model not loaded. 

To use llama-cpp-python:
1. Download a GGUF model from HuggingFace
2. Place it in ~/.hyperthymesia/models/
3. Set model path in config

Or use Ollama (easier): https://ollama.ai"""
            
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            return f"❌ llama-cpp error: {e}"
    
    def get_backend_info(self) -> Dict:
        """Get information about the active backend."""
        return {
            'backend': self.backend,
            'model': self.model_name,
            'available': self.is_available()
        }


# For BYOK (Bring Your Own Key) - Cloud fallback
class CloudLLM:
    """Cloud LLM fallback using user's API keys."""
    
    def __init__(self, provider: str = 'openai', api_key: Optional[str] = None):
        """
        Initialize cloud LLM.
        
        Args:
            provider: 'openai', 'anthropic', or 'gemini'
            api_key: User's API key
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f'{provider.upper()}_API_KEY')
        
        if not self.api_key:
            raise ValueError(f"No API key found for {provider}")
    
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return self.api_key is not None
    
    def answer_question(self, question: str, context: str,
                       max_tokens: int = 500) -> str:
        """Answer using cloud API."""
        
        if self.provider == 'openai':
            return self._call_openai(question, context, max_tokens)
        elif self.provider == 'anthropic':
            return self._call_anthropic(question, context, max_tokens)
        elif self.provider == 'gemini':
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
                    {"role": "system", "content": "Answer questions based on provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                max_tokens=max_tokens,
                temperature=0.7
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
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context:"
                }]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            return f"❌ Anthropic error: {e}"
    
    def _call_gemini(self, question: str, context: str, max_tokens: int) -> str:
        """Call Gemini API."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            
            response = model.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': 0.7
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            return f"❌ Gemini error: {e}"