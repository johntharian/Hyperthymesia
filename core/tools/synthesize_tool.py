"""
SynthesizeTool for agent to generate comprehensive answers.
"""
from typing import Dict, Any, List, Optional
from core.tools.base_tool import BaseTool, ToolResult
from core.local_llm import LocalLLM


class SynthesizeTool(BaseTool):
    """
    Tool for synthesizing information into coherent answers.

    Capabilities:
    - Combine search results with analysis
    - Generate explanations
    - Provide code examples
    - Create actionable recommendations
    """

    def __init__(self):
        """Initialize SynthesizeTool."""
        super().__init__(
            name='synthesize',
            description='''Synthesize information into a comprehensive answer.
Use this to combine search results, analysis, and context into a clear answer.
Parameters:
  - question (str, required): Original user question
  - search_results (list, optional): Results from search tool
  - analysis_results (list, optional): Results from analyze tool
  - context (str, optional): Additional context
Returns: Comprehensive answer with explanation and examples'''
        )
        self.llm = LocalLLM()

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute synthesis.

        Args:
            question (str): Original user question
            search_results (list): Search results to synthesize
            analysis_results (list): Analysis results to include
            context (str): Additional context

        Returns:
            ToolResult with synthesized answer
        """
        # Validate required params
        if not self.validate_params(['question'], kwargs):
            return ToolResult(
                success=False,
                data=None,
                message='Missing required parameter: question'
            )

        question = kwargs['question']
        search_results = kwargs.get('search_results', [])
        analysis_results = kwargs.get('analysis_results', [])
        context = kwargs.get('context', '')

        try:
            self.logger.debug(f"SynthesizeTool: Synthesizing answer for '{question}'")

            # If no LLM available, return basic synthesis
            if not self.llm.is_available():
                return self._synthesize_basic(
                    question, search_results, analysis_results, context
                )

            # Use LLM to synthesize
            return self._synthesize_with_llm(
                question, search_results, analysis_results, context
            )

        except Exception as e:
            self.logger.error(f"SynthesizeTool error: {e}")
            return ToolResult(
                success=False,
                data=None,
                message=f'Synthesis error: {str(e)}'
            )

    def _synthesize_basic(self, question: str, search_results: List,
                         analysis_results: List, context: str) -> ToolResult:
        """Basic synthesis without LLM."""
        answer_parts = []

        # Add context
        if context:
            answer_parts.append(f"Context: {context}\n")

        # Add search results
        if search_results:
            answer_parts.append("Relevant files found:")
            for i, result in enumerate(search_results[:3], 1):
                filename = result.get('filename', 'unknown')
                path = result.get('path', 'unknown')
                snippet = result.get('snippet', '')
                answer_parts.append(f"  {i}. {filename} ({path})")
                if snippet:
                    answer_parts.append(f"     Preview: {snippet[:100]}...")

        # Add analysis results
        if analysis_results:
            answer_parts.append("\nCode Analysis:")
            for result in analysis_results:
                if isinstance(result, dict):
                    for key, value in result.items():
                        answer_parts.append(f"  - {key}: {value}")

        answer = "\n".join(answer_parts) if answer_parts else "No results found."

        return ToolResult(
            success=True,
            data={'answer': answer, 'sources': search_results},
            message='Generated basic answer without LLM',
            metadata={'method': 'basic'}
        )

    def _synthesize_with_llm(self, question: str, search_results: List,
                            analysis_results: List, context: str) -> ToolResult:
        """Synthesis using local LLM for better quality."""
        # Build context for LLM
        llm_context = self._build_llm_context(
            question, search_results, analysis_results, context
        )

        # Create synthesis prompt
        prompt = f"""You are a code expert helping a developer understand their codebase.

User's question: {question}

Available information:
{llm_context}

Based on the information above, provide a clear, concise answer to the user's question.
Include:
1. Direct answer to their question
2. Relevant code examples or file references
3. Key patterns or concepts
4. Practical recommendations

Keep the answer focused and actionable."""

        try:
            answer = self.llm.answer_question(question, llm_context, max_tokens=500)

            return ToolResult(
                success=True,
                data={
                    'answer': answer,
                    'sources': search_results,
                    'analysis': analysis_results,
                },
                message='Generated answer using local LLM',
                metadata={'method': 'llm'}
            )

        except Exception as e:
            self.logger.warning(f"LLM synthesis failed, using basic: {e}")
            return self._synthesize_basic(question, search_results, analysis_results, context)

    def _build_llm_context(self, question: str, search_results: List,
                           analysis_results: List, context: str) -> str:
        """Build context string for LLM."""
        parts = []

        if context:
            parts.append(f"Context:\n{context}\n")

        if search_results:
            parts.append("Search Results:")
            for i, result in enumerate(search_results[:5], 1):
                filename = result.get('filename', 'unknown')
                snippet = result.get('snippet', '')
                score = result.get('score', 0)
                parts.append(f"  {i}. {filename} (relevance: {score:.2f})")
                if snippet:
                    parts.append(f"     {snippet[:150]}")

        if analysis_results:
            parts.append("\nCode Analysis:")
            for result in analysis_results:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (str, int, float)):
                            parts.append(f"  - {key}: {value}")
                        elif isinstance(value, list) and len(value) <= 5:
                            parts.append(f"  - {key}: {', '.join(str(v) for v in value)}")

        return "\n".join(parts)

    def compare_implementations(self, question: str, files: List[str]) -> ToolResult:
        """Compare implementations across multiple files."""
        comparison = {
            'question': question,
            'files': files,
            'comparison': 'Comparison not yet implemented'
        }

        return ToolResult(
            success=True,
            data=comparison,
            message='Comparison generated'
        )
