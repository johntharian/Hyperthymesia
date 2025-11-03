# Hyperthymesia Agentic Mode Architecture

## Overview

Hyperthymesia now includes an **agentic mode** that uses local LLM reasoning to intelligently process complex developer queries. Instead of always following a fixed pipeline (search → return results), the agent decides what actions to take based on understanding the query.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│          User Query / Command                        │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────▼──────────┐
        │ Query Analysis    │
        │ (complexity)      │
        └────────┬──────────┘
                 │
         ┌───────▼────────────┐
         │ Simple or Complex? │
         └───┬──────────┬─────┘
             │          │
        ┌────▼──┐   ┌───▼─────────────────────┐
        │Simple │   │ Complex → Agentic Mode  │
        │Search │   │                         │
        └───────┘   └───┬──────────────────┬──┘
                        │                  │
                   ┌────▼──────────────────▼────┐
                   │  HyperthymesiaAgent        │
                   ├────────────────────────────┤
                   │ 1. Reasoning (LLM thinks)  │
                   │ 2. Planning (creates plan) │
                   │ 3. Tool Selection          │
                   │ 4. Execution (use tools)   │
                   │ 5. Synthesis (answer)      │
                   └────┬──────────────────┬────┘
                        │                  │
            ┌───────────▼──────────────────▼───────────┐
            │         Tool Execution Engine            │
            ├──────────────────────────────────────────┤
            │ ┌──────────────┐ ┌──────────────┐        │
            │ │ SearchTool   │ │ AnalyzeTool  │        │
            │ ├──────────────┤ ├──────────────┤        │
            │ │ • Hybrid     │ │ • Extract    │        │
            │ │   search     │ │   functions  │        │
            │ │ • Concept    │ │ • Extract    │        │
            │ │   expansion  │ │   classes    │        │
            │ │ • Path       │ │ • Analyze    │        │
            │ │   filtering  │ │   imports    │        │
            │ └──────────────┘ └──────────────┘        │
            │                                          │
            │ ┌──────────────────────────────────────┐ │
            │ │ SynthesizeTool                       │ │
            │ ├──────────────────────────────────────┤ │
            │ │ • Combine search + analysis          │ │
            │ │ • LLM-generated answers              │ │
            │ │ • Code examples + explanations       │ │
            │ └──────────────────────────────────────┘ │
            └──────────────────────────────────────────┘
                        │
            ┌───────────▼──────────────┐
            │   Final Answer + Sources │
            │   Reasoning Chain        │
            │   (with thinking shown)  │
            └──────────────────────────┘
```

## Components

### 1. HyperthymesiaAgent (`core/agent.py`)

Main orchestrator that coordinates the agentic workflow.

**Key Methods:**
- `process_query(query, verbose)` - Main entry point
- `_generate_reasoning(query, analysis)` - Uses LLM to think about the problem
- `_generate_plan(query, reasoning)` - Creates step-by-step action plan
- `_extract_next_action(query, plan, steps)` - Determines next tool to use
- `_execute_tool(action)` - Runs selected tool
- `_synthesize_answer(query, steps)` - Creates final answer

**Workflow:**
1. Analyze query complexity
2. If simple → direct search (fast)
3. If complex → start agentic mode
4. Use LLM to reason about problem
5. Generate action plan
6. Execute tools in sequence
7. Synthesize comprehensive answer

### 2. Tool System (`core/tools/`)

Reusable components that the agent invokes.

#### BaseTool (`core/tools/base_tool.py`)
Abstract base class defining tool interface.

```python
class BaseTool(ABC):
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
```

**ToolResult:**
```python
@dataclass
class ToolResult:
    success: bool          # Did tool succeed?
    data: Any             # Result data
    message: str          # Human-readable message
    metadata: Dict        # Additional info
```

#### SearchTool (`core/tools/search_tool.py`)
Finds relevant code in indexed documents.

**Capabilities:**
- Hybrid search (keyword + semantic)
- Query expansion for abstract questions
- File type and path filtering
- Concept-based search

**Usage:**
```python
search_tool = SearchTool()
result = search_tool.execute(
    query="retry mechanism",
    limit=5,
    file_type=".py"
)
```

#### AnalyzeTool (`core/tools/analyze_tool.py`)
Analyzes code to extract structure and patterns.

**Capabilities:**
- Extract functions and classes
- Analyze imports
- Get code structure
- Find patterns

**Usage:**
```python
analyze_tool = AnalyzeTool()
result = analyze_tool.execute(
    file_path="core/search.py",
    analysis_type="functions"
)
```

#### SynthesizeTool (`core/tools/synthesize_tool.py`)
Generates comprehensive answers from information.

**Capabilities:**
- Combine search and analysis results
- Generate explanations with LLM
- Create actionable recommendations
- Compare implementations

**Usage:**
```python
synthesize_tool = SynthesizeTool()
result = synthesize_tool.execute(
    question="How do retries work?",
    search_results=[...],
    analysis_results=[...]
)
```

## Agent Flow Example

### User Query
```
"How do retries work in the codebase?"
```

### Agent Processing

**Step 1: Analysis**
```
Complexity: 5 (complex)
Is question: true
Has vague language: false
Strategy: agentic
```

**Step 2: Reasoning**
```
💭 Agent thinks:
- User wants to understand retry mechanism
- Need to search for retry-related code
- Need to analyze implementation
- Need to explain how it works
- Should look for patterns and examples
```

**Step 3: Planning**
```
📋 Action Plan:
1. search("retry exponential backoff timeout")
2. analyze(retry_file.py)
3. synthesize(results)
```

**Step 4: Execution**

```
🔧 Step 1: search()
   Found 4 files with retry patterns

🔧 Step 2: analyze()
   Found 2 functions: retry_with_backoff, exponential_backoff

🔧 Step 3: synthesize()
   Generated answer with code examples
```

**Step 5: Answer**
```
Retry mechanism in your codebase:

1. Core Implementation (retry.py):
   - exponential_backoff() function handles delay calculation
   - retry_with_backoff() decorator wraps functions

2. How it works:
   - Attempts action up to max_retries times
   - Waits between attempts with exponential backoff
   - Logs each attempt and failure

3. Usage example:
   @retry_with_backoff(max_retries=3)
   def api_call():
       ...
```

## Design Principles

### 1. Local-First
- All reasoning happens on-device
- No cloud calls required
- Complete privacy

### 2. Transparent
- Show thinking process to user
- Explain why certain tools were chosen
- Reasoning chain visible

### 3. Graceful Degradation
- Falls back to simple search if LLM unavailable
- Tools fail safely without crashing
- Basic synthesis works without LLM

### 4. Token-Aware
- Compresses results before passing to LLM
- Respects context window limits
- Efficient context building

### 5. Extensible
- Easy to add new tools
- Tool interface is simple
- Agent coordinates multiple tools

## Tool Integration

Tools are discovered and registered dynamically:

```python
agent = HyperthymesiaAgent()

# All tools automatically available
agent.tools = {
    'search': SearchTool(),
    'analyze': AnalyzeTool(),
    'synthesize': SynthesizeTool(),
}
```

Adding a new tool:

```python
class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name='my_tool',
            description='What this tool does'
        )

    def execute(self, **kwargs) -> ToolResult:
        # Implementation
        return ToolResult(...)

# Register in agent
agent.tools['my_tool'] = MyTool()
```

## LLM Requirements

The agent works with:
- **Ollama** (local, easiest)
- **MLX** (Mac, Apple Silicon optimized)
- **llama-cpp-python** (cross-platform)

Fallback to basic synthesis if LLM unavailable.

## Query Complexity Routing

```python
if query_is_simple():
    # Fast path: direct search
    search("query")
else:
    # Smart path: agentic reasoning
    agent.process_query("query")
```

**Simple queries:**
- "config" → direct search
- "auth.py" → filename search
- "api" → keyword search

**Complex queries:**
- "how do retries work?" → agentic
- "where is error handling used?" → agentic
- "compare logging in these two files" → agentic

## Performance Considerations

### Latency
- Simple query: ~200-300ms (search only)
- Complex query: ~2-5 seconds (reasoning + search + synthesis)

### Token Usage
- Reasoning: ~100-150 tokens
- Planning: ~50-100 tokens
- Analysis: ~100-200 tokens per file
- Total: ~300-600 tokens per query

### Memory
- Agent: ~50MB (LLM loaded)
- Tools: ~10MB
- Total: ~60MB additional

## Testing

The agentic mode includes:
- Unit tests for each tool
- Integration tests for agent coordination
- End-to-end scenario tests
- Reasoning clarity tests

```bash
# Test tools
pytest tests/test_tools.py -v

# Test agent
pytest tests/test_agent.py -v

# Test scenarios
pytest tests/test_agent_scenarios.py -v
```

## Future Enhancements

1. **Multi-file analysis**
   - Compare implementations across files
   - Track dependencies

2. **Interactive refinement**
   - Agent asks clarifying questions
   - User provides feedback
   - Agent iterates

3. **Caching**
   - Cache frequent searches
   - Reuse analysis results
   - Faster subsequent queries

4. **Custom tools**
   - User-defined tools
   - Domain-specific analyzers
   - Plugin system

5. **Conversation history**
   - Multi-turn conversations
   - Context carried across queries
   - Learning from interactions

## Examples

### Example 1: "How do retries work?"
```
Query: "How do retries work?"
       ↓
    Agent decides: Search → Analyze → Synthesize
       ↓
    Finds: 3 retry-related files
       ↓
    Analyzes: exponential_backoff function
       ↓
    Returns: Detailed explanation with code examples
```

### Example 2: "Where is authentication handled?"
```
Query: "Where is authentication handled?"
       ↓
    Agent decides: Search for auth patterns
       ↓
    Finds: auth.py, login function, Session class
       ↓
    Analyzes: Authentication flow
       ↓
    Returns: File locations + implementation details
```

### Example 3: "What's the difference between two error handlers?"
```
Query: "What's the difference between two error handlers?"
       ↓
    Agent decides: Search → Analyze both → Compare → Synthesize
       ↓
    Returns: Side-by-side comparison with pros/cons
```

## Summary

The agentic mode transforms Hyperthymesia from a simple search tool into an intelligent code assistant that:

1. **Understands intent** - Not just keywords
2. **Reasons about problems** - Multi-step thinking
3. **Plans actions** - Coordinates multiple tools
4. **Executes intelligently** - Adapts based on results
5. **Synthesizes answers** - Comprehensive, not just results

All entirely **local and private**.

---

**Status**: Core agentic framework complete. Ready for CLI integration and testing.
