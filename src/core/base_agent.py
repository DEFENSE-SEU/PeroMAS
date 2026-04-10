"""
Abstract Base Agent for PSC_Agents.

The "Controller" that orchestrates LLM reasoning and tool execution.
Serves as the parent class for all LangGraph nodes.

Author: PSC_Agents Team
"""

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Ensure core directory is in path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LLMConfig, MCPConfig, Settings
from llm import LLMClient
from tool import MCPToolRegistry


class BaseAgent(ABC):
    """
    Abstract base class for all LangGraph agent nodes.

    Orchestrates the LLM (Brain) and Tool Registry (Hands) to perform
    autonomous reasoning and action execution.

    Attributes:
        name: Human-readable name of the agent.
        settings: Aggregated settings for LLM and MCP.
        registry: MCPToolRegistry for tool management.
        llm: LLMClient for LLM interactions.
        logger: Logger instance.

    Example:
        >>> class ResearchAgent(BaseAgent):
        ...     async def run(self, state: dict) -> dict:
        ...         result = await self.autonomous_thinking(
        ...             prompt="Find papers about perovskite",
        ...             state=state
        ...         )
        ...         return {"research_results": result}
        ...
        >>> async with ResearchAgent("researcher", settings) as agent:
        ...     output = await agent.run({"query": "perovskite"})
    """

    def __init__(
        self,
        name: str,
        settings: Settings | None = None,
        llm_config: LLMConfig | None = None,
        mcp_config: MCPConfig | None = None,
        max_tool_output_length: int = 5000,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            name: Human-readable name for the agent.
            settings: Complete Settings object. If provided, overrides individual configs.
            llm_config: LLM configuration (used if settings not provided).
            mcp_config: MCP configuration (used if settings not provided).
            max_tool_output_length: Maximum length of tool output before truncation (default: 5000).
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.max_tool_output_length = max_tool_output_length

        # Use settings if provided, otherwise build from individual configs
        if settings:
            self.settings = settings
        else:
            self.settings = Settings(
                llm=llm_config or LLMConfig(),
                mcp=mcp_config or MCPConfig(),
            )

        # Initialize components (lazy - actual connection in __aenter__)
        self.registry = MCPToolRegistry(self.settings.mcp)
        self.llm: LLMClient | None = None
        self._initialized = False

        self.logger.info(f"Agent '{name}' created")

    async def _initialize(self) -> None:
        """Initialize LLM client and connect to MCP servers."""
        if self._initialized:
            return

        # Initialize LLM client
        if self.settings.llm.is_valid():
            self.llm = LLMClient(self.settings.llm)
            self.logger.info("LLM client initialized")
        else:
            self.logger.warning(
                "LLM configuration invalid - running without LLM capability"
            )

        # Connect to MCP servers
        if self.settings.mcp.servers:
            results = await self.registry.initialize()
            connected = sum(results.values())
            self.logger.info(f"Connected to {connected}/{len(results)} MCP servers")

        self._initialized = True

    async def _shutdown(self) -> None:
        """Shutdown connections."""
        if self.registry.is_initialized():
            await self.registry.shutdown()
        self._initialized = False
        self.logger.info(f"Agent '{self.name}' shutdown complete")

    def _get_system_prompt(
        self,
        state: dict[str, Any],
        default_prompt: str | None = None,
    ) -> str | None:
        """
        Get the system prompt for the current reasoning session.

        Subclasses can override this method to dynamically generate system prompts
        based on the current state. For example:
        - "You are a data analyst..." for analysis tasks
        - "You are a writing expert..." for editing tasks

        Args:
            state: Current state dictionary.
            default_prompt: Default prompt passed to autonomous_thinking.

        Returns:
            System prompt string or None.

        Example:
            >>> class AnalystAgent(BaseAgent):
            ...     def _get_system_prompt(self, state, default_prompt):
            ...         task_type = state.get('task_type', 'analysis')
            ...         if task_type == 'analysis':
            ...             return "You are a data analyst expert..."
            ...         elif task_type == 'writing':
            ...             return "You are a writing expert..."
            ...         return default_prompt
        """
        return default_prompt

    def _truncate_tool_output(self, output: str, tool_name: str = "") -> str:
        """
        Truncate tool output if it exceeds max_tool_output_length.
        
        Only truncates large unstructured content (e.g., paper full text).
        Important structured outputs are preserved.

        Args:
            output: Raw tool output string.
            tool_name: Name of the tool that produced this output.

        Returns:
            Truncated output with warning if needed.
        """
        # Allow subclasses to process specific tool outputs
        processed = self._process_tool_output(output, tool_name)
        if processed is not None:
            return processed
        
        # Tools that produce large unstructured content - OK to truncate
        truncatable_tools = {
            "read_paper",       # Paper full text
            "download_paper",   # Binary/large content  
            "search_papers",    # Many search results
        }
        
        # Don't truncate important structured outputs from other tools
        if tool_name and tool_name not in truncatable_tools:
            return output
        
        if len(output) <= self.max_tool_output_length:
            return output

        truncated = output[: self.max_tool_output_length]
        original_length = len(output)
        truncated_length = original_length - self.max_tool_output_length
        warning = (
            f"\n\n...(truncated {truncated_length} characters, "
            f"original length: {original_length})"
        )

        self.logger.warning(
            f"Tool output truncated: {original_length} -> {self.max_tool_output_length} chars"
        )

        return truncated + warning

    def _process_tool_output(self, output: str, tool_name: str) -> str | None:
        """
        Hook for subclasses to process specific tool outputs.
        
        Override this method to handle tool-specific output processing
        (e.g., caching large binary data).
        
        Returns:
            Processed output string, or None to use default truncation.
        """
        return None

    def _preprocess_tool_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Hook for subclasses to preprocess tool arguments before execution.
        
        Override this method to inject cached data or modify arguments.
        
        Returns:
            Modified arguments dict.
        """
        return args

    async def autonomous_thinking(
        self,
        prompt: str,
        state: dict[str, Any],
        system_message: str | None = None,
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Execute a ReAct-style reasoning loop.

        The agent will:
        1. Get available tools from the registry
        2. Ask the LLM to reason about the task
        3. If the LLM requests tool calls, execute them
        4. Feed results back to the LLM
        5. Repeat until done or max iterations reached

        Args:
            prompt: The task/question for the agent.
            state: Current state dict (for context).
            system_message: Optional system prompt.
            max_iterations: Maximum tool-calling iterations.

        Returns:
            Dictionary containing:
            - 'response': Final LLM response text
            - 'tool_calls': List of tool calls made
            - 'tool_results': Results from tool executions
            - 'iterations': Number of iterations performed

        Example:
            >>> result = await agent.autonomous_thinking(
            ...     prompt="Search for recent papers on perovskite stability",
            ...     state={"context": "materials research"},
            ...     max_iterations=5
            ... )
        """
        if not self.llm:
            self.logger.error("LLM client not available")
            return {
                "response": "[ERROR] LLM not configured",
                "tool_calls": [],
                "tool_results": [],
                "iterations": 0,
            }

        # Get tool schemas
        tools = await self.registry.get_tools_schema() if self.registry.is_initialized() else []
        
        # Debug: Print tool info
        self.logger.info(f"Registry initialized: {self.registry.is_initialized()}")
        self.logger.info(f"Available tools: {len(tools)}")
        if tools:
            tool_names = [t.get('function', {}).get('name', 'unknown') for t in tools]
            self.logger.info(f"Tool names: {tool_names}")

        # Build initial messages
        messages: list[dict[str, Any]] = []
        
        # Use _get_system_prompt to allow dynamic system prompt
        final_system_prompt = self._get_system_prompt(state, system_message)
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})

        # Add context from state if available
        context_str = ""
        if state:
            context_str = f"\n\nCurrent context:\n{state}"

        messages.append({"role": "user", "content": prompt + context_str})

        # Track results
        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []
        iterations = 0
        response = None  # Initialize response
        
        # Track consecutive tool calls for deduplication
        _last_tool_name: str | None = None
        _consecutive_count: int = 0

        # ReAct loop
        while iterations < max_iterations:
            iterations += 1
            self.logger.debug(f"Thinking iteration {iterations}")

            # Invoke LLM (streaming for real-time output)
            response = await self.llm.ainvoke_streaming(
                messages, tools=tools if tools else None, print_stream=True,
            )

            # Check for tool calls
            if not self.llm.has_tool_calls(response):
                # No tool calls - we're done
                self.logger.debug("No tool calls, finishing")
                break

            # Process tool calls
            tool_calls = self.llm.get_tool_calls(response)
            messages.append(response)  # Add assistant message with tool calls

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                self.logger.info(f"Executing tool: {tool_name}")
                
                # Tool call visualization with de-duplication.
                if tool_name == _last_tool_name:
                    _consecutive_count += 1
                    # Update the same line with the call count.
                    print(f"\r   🔄 [{self.name}] {tool_name} called {_consecutive_count}x (consecutive)", end="", flush=True)
                else:
                    if _last_tool_name is not None and _consecutive_count > 1:
                        print()  # End the previous tool's counter line.
                    _consecutive_count = 1
                    _last_tool_name = tool_name
                    print(f"\n🔧 [{self.name}] Calling Tool: {tool_name}")
                    print(f"   📥 Arguments: {str(tool_args)[:200]}{'...' if len(str(tool_args)) > 200 else ''}")
                
                all_tool_calls.append(tc)

                # Allow subclasses to preprocess arguments
                tool_args = self._preprocess_tool_args(tool_name, tool_args)

                try:
                    result = await self.registry.call_tool(tool_name, tool_args)
                    result_str = str(result) if result else "No result"
                    # Truncate if too long to avoid context window overflow
                    result_str = self._truncate_tool_output(result_str, tool_name)
                except Exception as e:
                    self.logger.error(f"Tool execution failed: {e}")
                    result_str = f"[ERROR] {e}"

                all_tool_results.append({
                    "tool": tool_name,
                    "result": result_str,
                })
                
                # Tool result visualization (details only for first call).
                if _consecutive_count == 1:
                    result_preview = result_str[:150] if len(result_str) > 150 else result_str
                    print(f"   📤 Result: {result_preview}{'...' if len(result_str) > 150 else ''}")

                # Add tool result message
                tool_message = self.llm.create_tool_message(tool_id, result_str)
                messages.append(tool_message)

        # Get final response content
        final_response = ""
        if response and hasattr(response, "content"):
            final_response = response.content or ""

        return {
            "response": final_response,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "iterations": iterations,
        }

    async def simple_invoke(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> str:
        """
        Simple LLM invocation without tools.

        Args:
            prompt: User prompt.
            system_message: Optional system message.

        Returns:
            LLM response as string.
        """
        if not self.llm:
            return "[ERROR] LLM not configured"

        return await self.llm.ainvoke_simple(prompt, system_message)

    @abstractmethod
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the agent's main logic (LangGraph contract).

        This is the core method for LangGraph integration. Subclasses
        MUST implement this to process the global state and return
        a partial state update.

        Args:
            state: The current global state dictionary from LangGraph.

        Returns:
            A dictionary containing the state fields to update.
            This will be merged into the global state by LangGraph.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement run()")

    async def __aenter__(self) -> "BaseAgent":
        """Async context manager entry - initializes connections."""
        await self._initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - shuts down connections."""
        await self._shutdown()

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"initialized={self._initialized})"
        )
