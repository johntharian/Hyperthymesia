"""
LLM integration for query understanding and rewriting.
"""

import os
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class LLMQueryRewriter:
    """Uses LLM to understand and rewrite complex queries."""

    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        """
        Initialize LLM rewriter.

        Args:
            api_key: API key for LLM provider (uses env var if None)
            provider: LLM provider ('openai', 'anthropic', or 'local')
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("LLM_API_KEY")

        if provider == "openai" and not self.api_key:
            raise ValueError(
                "LLM API key not found. Set LLM_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "gemini":
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel("gemini-2.0-flash")
            except:
                raise Exception("Failed to initialize Gemini client")
        elif self.provider == "openai":
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai"
                )
        elif self.provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Run: pip install anthropic"
                )
        # Add more providers as needed

    def rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Rewrite a complex query into effective search terms.

        Args:
            query: Original user query
            context: Optional context about previous searches

        Returns:
            Rewritten query optimized for search
        """
        prompt = self._build_rewrite_prompt(query, context)

        try:
            if self.provider == "gemini":
                return self._call_gemini(prompt)
            elif self.provider == "openai":
                return self._call_openai(prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt)
            else:
                # Fallback: return original query
                return query
        except Exception as e:
            print(f"LLM rewrite failed: {e}")
            # Fallback to original query
            return query

    def _build_rewrite_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build the prompt for query rewriting."""
        prompt = """You are a search query optimizer for a personal document search system.

Your task: Transform the user's natural language query into 2-5 effective search keywords.

Guidelines:
- Extract the core concepts and remove filler words
- Keep technical terms and specific names
- Convert questions into keywords
- Remove vague references like "that", "something"
- If time is mentioned, include it as a keyword
- Be concise - return only the optimized keywords

Examples:
"that document about machine learning I read last month" → "machine learning document"
"how do I implement async functions in python?" → "python async functions implementation"
"show me files related to the Chicago office project" → "Chicago office project"
"what's the latest on neural networks?" → "neural networks latest"
"the PDF with the tutorial" → "PDF tutorial"

"""

        if context:
            prompt += f"\nPrevious context: {context}\n"

        prompt += f'\nUser query: "{query}"\n\nOptimized search keywords:'

        return prompt

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        response = self.client.generate_content(prompt)
        rewritten = response.text.strip()
        rewritten = rewritten.strip("\"'")
        return rewritten

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap
            messages=[
                {"role": "system", "content": "You are a search query optimizer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.3,  # Lower temperature for more consistent results
        )

        rewritten = response.choices[0].message.content.strip()
        # Remove quotes if LLM added them
        rewritten = rewritten.strip("\"'")
        return rewritten

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        message = self.client.messages.create(
            model="claude-3-haiku-20240307",  # Fast and cheap
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )

        rewritten = message.content[0].text.strip()
        rewritten = rewritten.strip("\"'")
        return rewritten

    def is_available(self) -> bool:
        """Check if LLM is available and configured."""
        return self.api_key is not None


# Singleton instance
_rewriter = None


def get_rewriter(provider: str = "gemini") -> Optional[LLMQueryRewriter]:
    """
    Get or create LLMQueryRewriter instance.

    Returns None if LLM is not configured (missing API key).
    """
    global _rewriter
    if _rewriter is None:
        try:
            _rewriter = LLMQueryRewriter(provider=provider)
        except ValueError:
            # API key not found - LLM features disabled
            return None
    return _rewriter
