"""
HyperthymesiaAgent - Main orchestrator for agentic query processing.
Routes complex queries through reasoning, planning, and tool execution.
Also enhances answers with RAG-powered detailed explanations.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from core.local_llm import LocalLLM
from core.query_analyzer import QueryAnalyzer
from core.rag_retriever import RAGRetriever
from core.tools import SearchTool, AnalyzeTool, SynthesizeTool
from core.tools.base_tool import ToolResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentAction:
    """An action the agent decides to take."""
    tool_name: str
    params: Dict[str, Any]
    reasoning: str


@dataclass
class AgentStep:
    """A step in the agent's reasoning process."""
    step_num: int
    description: str
    action: Optional[AgentAction] = None
    result: Optional[ToolResult] = None


@dataclass
class AgentResponse:
    """Final response from the agent."""
    answer: str
    sources: List[str]
    steps: List[AgentStep]
    reasoning_chain: str
    success: bool
    detailed_explanation: Optional[str] = None
    explanation_sources: Optional[List[str]] = None


class HyperthymesiaAgent:
    """
    Intelligent agent that orchestrates tool use for complex queries.

    Flow:
    1. Analyze query complexity
    2. Generate reasoning about what to do
    3. Create action plan
    4. Execute tools
    5. Synthesize answer
    """

    def __init__(self):
        """Initialize the agent with tools and LLM."""
        self.llm = LocalLLM()
        self.analyzer = QueryAnalyzer()
        self.rag_retriever = RAGRetriever()
        self.tools = {
            'search': SearchTool(),
            'analyze': AnalyzeTool(),
            'synthesize': SynthesizeTool(),
        }
        self.logger = logger
        self.max_steps = 5  # Prevent infinite loops
        self.max_tokens_per_step = 500
        self.enable_rag_enhancement = True  # Enable RAG explanations

    def process_query(self, query: str, verbose: bool = False) -> AgentResponse:
        """
        Process a user query using agentic reasoning.

        Args:
            query: User's question or query
            verbose: If True, show reasoning steps

        Returns:
            AgentResponse with answer and reasoning
        """
        self.logger.info(f"Agent processing: {query}")

        # Step 1: Analyze query
        analysis = self.analyzer.analyze(query)
        self.logger.debug(f"Query analysis: {analysis}")

        # Step 2: Decide if agentic approach needed
        if not analysis['is_complex']:
            self.logger.debug("Simple query, using direct search")
            return self._handle_simple_query(query)

        # Step 3: Start agentic reasoning
        self.logger.debug("Complex query, starting agentic processing")
        steps = []
        reasoning_chain = ""

        try:
            # Generate reasoning
            reasoning = self._generate_reasoning(query, analysis)
            reasoning_chain = reasoning
            if verbose:
                print(f"\n💭 Agent Reasoning:\n{reasoning}\n")

            # Generate plan
            plan = self._generate_plan(query, reasoning)
            if verbose:
                print(f"📋 Plan:\n{plan}\n")

            # Execute plan
            step_num = 0
            for step_num in range(self.max_steps):
                action = self._extract_next_action(query, plan, steps)

                if not action:
                    break  # No more actions

                step = AgentStep(
                    step_num=step_num + 1,
                    description=f"Execute: {action.tool_name}",
                    action=action
                )

                # Execute tool
                result = self._execute_tool(action)
                step.result = result

                steps.append(step)

                if verbose:
                    print(f"🔧 Step {step_num + 1}: {action.tool_name}")
                    print(f"   Reasoning: {action.reasoning}")
                    print(f"   Result: {result.message}\n")

                # Check if we have enough information
                if self._should_synthesize(steps, result):
                    break

            # Synthesize final answer
            final_response = self._synthesize_answer(query, steps)

            # Enhance with RAG explanation if enabled
            detailed_explanation = None
            explanation_sources = None

            if self.enable_rag_enhancement:
                if verbose:
                    print("📚 Retrieving detailed context for explanation...\n")

                try:
                    rag_result = self._enhance_with_rag(query)
                    detailed_explanation = rag_result['explanation']
                    explanation_sources = rag_result['sources']
                except Exception as e:
                    self.logger.debug(f"RAG enhancement failed: {e}")
                    # Continue without RAG enhancement

            return AgentResponse(
                answer=final_response,
                sources=self._extract_sources(steps),
                steps=steps,
                reasoning_chain=reasoning_chain,
                success=True,
                detailed_explanation=detailed_explanation,
                explanation_sources=explanation_sources
            )

        except Exception as e:
            self.logger.error(f"Agent error: {e}")
            return AgentResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                steps=steps,
                reasoning_chain=reasoning_chain,
                success=False
            )

    def _handle_simple_query(self, query: str) -> AgentResponse:
        """Handle simple queries with direct search."""
        search_tool = self.tools['search']
        result = search_tool.execute(query=query, limit=5)

        if result.success and result.data:
            answer = f"Found {len(result.data)} relevant files:\n"
            for i, item in enumerate(result.data[:3], 1):
                answer += f"{i}. {item['filename']} - {item['snippet'][:100]}...\n"
        else:
            answer = "No results found."

        return AgentResponse(
            answer=answer,
            sources=[r.get('path', '') for r in result.data if result.data],
            steps=[],
            reasoning_chain="Direct search (simple query)",
            success=result.success
        )

    def _generate_reasoning(self, query: str, analysis: Dict) -> str:
        """Generate reasoning about the query using LLM."""
        if not self.llm.is_available():
            return self._simple_reasoning(query, analysis)

        prompt = f"""You are analyzing a developer's code question to decide what to do.

Question: {query}

Analysis:
- Is complex: {analysis['is_complex']}
- Is question: {analysis['is_question']}
- Has vague language: {analysis['has_vague_language']}
- Complexity score: {analysis['complexity_score']}

Think step by step:
1. What is the developer really asking?
2. What information do I need to answer this?
3. What tools should I use? (search, analyze, synthesize)
4. In what order?

Reasoning:"""

        try:
            response = self.llm.generate(prompt, max_tokens=self.max_tokens_per_step)
            return response.strip()
        except:
            return self._simple_reasoning(query, analysis)

    def _simple_reasoning(self, query: str, analysis: Dict) -> str:
        """Simple reasoning without LLM."""
        reasoning = f"Query: {query}\n"
        reasoning += f"Complexity: {analysis['complexity_score']}/10\n"

        if analysis['has_vague_language']:
            reasoning += "- Has vague language, need to search broadly\n"

        if analysis['is_question']:
            reasoning += "- Is a question, need detailed analysis\n"

        reasoning += "Plan: Search for relevant code, then analyze patterns"
        return reasoning

    def _generate_plan(self, query: str, reasoning: str) -> str:
        """Generate action plan using LLM."""
        if not self.llm.is_available():
            return "1. search(query)\n2. synthesize(results)"

        prompt = f"""Based on this reasoning about: {query}

Reasoning: {reasoning}

Create a step-by-step action plan using these tools:
- search(query): Find relevant code
- analyze(file_path): Understand code structure
- synthesize(question, results): Create answer

Plan (numbered steps, one per line):"""

        try:
            response = self.llm.generate(prompt, max_tokens=self.max_tokens_per_step)
            return response.strip()
        except:
            return "1. search(query)\n2. synthesize(results)"

    def _extract_next_action(self, query: str, plan: str, steps: List[AgentStep]) -> Optional[AgentAction]:
        """Extract the next action from the plan."""
        lines = plan.split('\n')

        for i, line in enumerate(lines):
            # Check if this step has been done
            step_num = i + 1
            if any(s.step_num == step_num for s in steps):
                continue

            # Parse action
            line = line.strip()
            if not line:
                continue

            # Extract tool name and params
            if 'search' in line.lower():
                return AgentAction(
                    tool_name='search',
                    params={'query': query, 'limit': 5},
                    reasoning=line
                )
            elif 'analyze' in line.lower():
                # Get file from previous search results
                if steps and steps[-1].result and steps[-1].result.data:
                    file_path = steps[-1].result.data[0].get('path', '')
                    if file_path:
                        return AgentAction(
                            tool_name='analyze',
                            params={'file_path': file_path, 'analysis_type': 'structure'},
                            reasoning=line
                        )
            elif 'synthesize' in line.lower():
                # Collect results from previous steps
                search_results = []
                analysis_results = []

                for step in steps:
                    if step.action.tool_name == 'search' and step.result.data:
                        search_results.extend(step.result.data)
                    elif step.action.tool_name == 'analyze' and step.result.data:
                        analysis_results.append(step.result.data)

                return AgentAction(
                    tool_name='synthesize',
                    params={
                        'question': query,
                        'search_results': search_results,
                        'analysis_results': analysis_results,
                    },
                    reasoning=line
                )

        return None

    def _execute_tool(self, action: AgentAction) -> ToolResult:
        """Execute a tool with given parameters."""
        self.logger.debug(f"Executing tool: {action.tool_name}")

        tool = self.tools.get(action.tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                message=f"Unknown tool: {action.tool_name}"
            )

        result = tool.execute(**action.params)
        return result

    def _should_synthesize(self, steps: List[AgentStep], last_result: ToolResult) -> bool:
        """Decide if we have enough information to synthesize."""
        if not steps:
            return False

        # If last step was successful and is synthesize, we're done
        if steps and steps[-1].action.tool_name == 'synthesize':
            return last_result.success

        # If we have search results, we can synthesize
        if any(s.action.tool_name == 'search' and s.result.success for s in steps):
            return True

        return False

    def _synthesize_answer(self, query: str, steps: List[AgentStep]) -> str:
        """Synthesize final answer from tool results."""
        search_results = []
        analysis_results = []

        for step in steps:
            if step.result and step.result.success:
                if step.action.tool_name == 'search':
                    search_results.extend(step.result.data or [])
                elif step.action.tool_name == 'analyze':
                    analysis_results.append(step.result.data)
                elif step.action.tool_name == 'synthesize':
                    # Use synthesize tool result directly
                    if isinstance(step.result.data, dict):
                        return step.result.data.get('answer', 'No answer generated')

        # Fallback synthesis
        if search_results:
            answer = f"Based on my analysis, here are the relevant files:\n\n"
            for i, result in enumerate(search_results[:3], 1):
                filename = result.get('filename', 'unknown')
                snippet = result.get('snippet', '')[:100]
                answer += f"{i}. **{filename}**\n"
                if snippet:
                    answer += f"   {snippet}...\n"
            return answer

        return "No relevant information found."

    def _extract_sources(self, steps: List[AgentStep]) -> List[str]:
        """Extract source files from tool results."""
        sources = []

        for step in steps:
            if step.result and step.result.success and step.result.data:
                if step.action.tool_name == 'search' and isinstance(step.result.data, list):
                    for item in step.result.data:
                        path = item.get('path', '')
                        if path and path not in sources:
                            sources.append(path)

        return sources

    def _enhance_with_rag(self, query: str) -> Dict[str, Any]:
        """
        Enhance agent answer with RAG-powered detailed explanation.

        Uses RAG retriever to find relevant context and generate
        a comprehensive explanation.

        Args:
            query: Original user query

        Returns:
            Dictionary with 'explanation' and 'sources' keys
        """
        self.logger.debug(f"Enhancing with RAG for: {query}")

        try:
            # Retrieve context using RAG
            result = self.rag_retriever.retrieve_context(query, num_chunks=5)

            if not result['context']:
                self.logger.debug("No RAG context found")
                return {
                    'explanation': None,
                    'sources': []
                }

            # Generate detailed explanation using the context
            if not self.llm.is_available():
                # Fallback: just use context as explanation (but still answer the question)
                explanation = self._build_context_explanation(result, query)
            else:
                # Use LLM to generate comprehensive explanation answering the question
                explanation = self._generate_rag_explanation(query, result['context'])

            # Extract sources
            sources = [s['path'] for s in result['sources']]

            self.logger.debug(f"RAG enhancement complete: {len(sources)} sources")

            return {
                'explanation': explanation,
                'sources': sources
            }

        except Exception as e:
            self.logger.error(f"RAG enhancement error: {e}")
            return {
                'explanation': None,
                'sources': []
            }

    def _generate_rag_explanation(self, query: str, context: str) -> str:
        """
        Generate detailed explanation using LLM and retrieved context.

        Answers the user's specific question based on the retrieved code context.

        Args:
            query: Original query/question
            context: Retrieved context from RAG

        Returns:
            Detailed explanation answering the question
        """
        prompt = f"""You are a senior software engineer. Answer the developer's question using the provided code context.

Developer's Question: {query}

Code Context (retrieved from the codebase):
{context}

Instructions:
1. Answer the question directly and comprehensively
2. Use specific code references from the context
3. Explain implementation details with examples
4. Show how different components work together
5. Be technical but clear
6. If the context doesn't fully answer the question, explain what you found and what's missing
7. Keep the explanation focused on what the developer asked

Answer:"""

        try:
            explanation = self.llm.generate(prompt, max_tokens=800)
            return explanation.strip()
        except Exception as e:
            self.logger.error(f"LLM explanation generation failed: {e}")
            return self._build_context_explanation({'context': context, 'sources': []}, query)

    def _build_context_explanation(self, rag_result: Dict, query: str = "") -> str:
        """
        Build explanation from context without LLM.

        Fallback when LLM is not available. Still answers the question
        based on retrieved context.

        Args:
            rag_result: RAG retrieval result
            query: Optional user query for context

        Returns:
            Explanation string
        """
        context = rag_result.get('context', '')
        sources = rag_result.get('sources', [])

        explanation = ""

        if query:
            explanation += f"## Answer to: {query}\n\n"

        explanation += "### Code Context\n\n"

        if context:
            explanation += context[:1000]  # Limit to first 1000 chars
            if len(context) > 1000:
                explanation += "\n\n[... additional context available ...]"
        else:
            explanation += "[No specific code context found]"

        if sources:
            explanation += "\n\n### Source Files\n"
            for source in sources[:5]:
                filename = source.get('file', 'unknown')
                path = source.get('path', 'unknown')
                explanation += f"- **{filename}** (`{path}`)\n"

        return explanation


# Singleton instance
_agent = None

def get_agent() -> HyperthymesiaAgent:
    """Get or create HyperthymesiaAgent instance."""
    global _agent
    if _agent is None:
        _agent = HyperthymesiaAgent()
    return _agent
