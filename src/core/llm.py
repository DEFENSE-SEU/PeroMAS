"""
LLM Client Wrapper for PSC_Agents.

Provides a unified interface for LLM interactions with support for
multiple providers and tool binding.
This module serves as the "Brain" of the agent system.

Based on Code2MCP's architecture:
- OpenAI/DeepSeek/Qwen/Google (via proxy): Use ChatOpenAI (OpenAI-compatible)
- Anthropic: Use ChatAnthropic (native SDK) 
- Ollama: Use ChatOllama (local)

Key insight: When using a proxy API (like gptsapi.net), ALL providers
including Gemini and Claude can use ChatOpenAI because the proxy
translates requests to OpenAI format.

Author: PSC_Agents Team
"""

import logging
import os
import time
import random
from typing import Any, AsyncIterator

# LangChain imports - wrapped for graceful degradation
try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False
    BaseMessage = None  # type: ignore
    AIMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    ToolMessage = None  # type: ignore

# Provider-specific imports
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None  # type: ignore

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ChatOllama = None  # type: ignore

from config import LLMConfig


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Client selection logic (based on Code2MCP):
    - anthropic (native API only): ChatAnthropic
    - ollama: ChatOllama  
    - Everything else (openai, deepseek, qwen, google via proxy): ChatOpenAI
    
    When using a proxy API, all requests go through ChatOpenAI because
    the proxy handles the translation to provider-specific formats.

    Attributes:
        config: LLMConfig containing API settings.
        llm: The underlying LangChain chat model instance.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize the LLM client.

        Args:
            config: LLMConfig with API settings.

        Raises:
            RuntimeError: If required LangChain packages are not installed.
            ValueError: If configuration is invalid.
        """
        if not LANGCHAIN_CORE_AVAILABLE:
            raise RuntimeError(
                "LangChain core is not installed. Install with: pip install langchain-core"
            )

        if not config.is_valid():
            raise ValueError(
                "Invalid LLM configuration. Ensure api_key and model_name are set."
            )

        self.config = config
        self.logger = logging.getLogger(f"{__name__}.LLMClient")
        
        # Statistics tracking
        self.total_calls = 0
        self.failed_calls = 0
        self.retry_count = 0

        # Create appropriate LLM client based on provider and whether using proxy
        self.llm = self._create_client()
        
        self.logger.info(
            f"LLMClient initialized: provider={config.provider}, model={config.model_name}, "
            f"uses_proxy={config.uses_proxy()}"
        )

    def _is_claude_model(self) -> bool:
        """Check if current model is a Claude model (requires special message format)."""
        model_name = self.config.model_name.lower()
        return "claude" in model_name or self.config.provider.lower() == "anthropic"

    def _create_client(self) -> Any:
        """
        Create the appropriate LangChain client.
        
        Logic (following Code2MCP pattern):
        1. If provider is 'anthropic' AND using native API -> ChatAnthropic
        2. If provider is 'ollama' -> ChatOllama
        3. Everything else (including proxy APIs) -> ChatOpenAI
        
        This works because:
        - Proxy APIs (like gptsapi.net) are OpenAI-compatible
        - DeepSeek/Qwen native APIs are also OpenAI-compatible
        
        Returns:
            LangChain chat model instance.
        """
        provider = self.config.provider.lower()
        uses_proxy = self.config.uses_proxy()
        
        self.logger.debug(f"Creating client for provider={provider}, uses_proxy={uses_proxy}")
        
        # Anthropic with native API (not proxy)
        if provider == "anthropic" and not uses_proxy:
            return self._create_anthropic_client()
        
        # Ollama (always local)
        if provider == "ollama":
            return self._create_ollama_client()
        
        # Everything else uses ChatOpenAI (OpenAI-compatible)
        # This includes:
        # - OpenAI native
        # - DeepSeek (native or proxy)
        # - Qwen (native or proxy)  
        # - Google/Gemini (via proxy only)
        # - Anthropic (via proxy)
        return self._create_openai_client()

    def _create_openai_client(self) -> Any:
        """
        Create ChatOpenAI client.
        
        Works for:
        - OpenAI (GPT models)
        - DeepSeek (OpenAI-compatible API)
        - Qwen (OpenAI-compatible API)
        - Any provider via OpenAI-compatible proxy
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "langchain-openai is not installed. Install with: pip install langchain-openai"
            )
        
        self.logger.info(
            f"Creating ChatOpenAI client: model={self.config.model_name}, "
            f"base_url={self.config.base_url}"
        )
        
        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            openai_api_key=self.config.api_key,
            openai_api_base=self.config.base_url,
            max_tokens=self.config.max_tokens,
            request_timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            streaming=False,
        )

    def _create_anthropic_client(self) -> Any:
        """
        Create ChatAnthropic client for Claude models with native API.
        
        Only used when:
        - provider is 'anthropic' 
        - AND using native Anthropic API (not a proxy)
        """
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "langchain-anthropic is not installed. Install with: pip install langchain-anthropic"
            )
        
        self.logger.info(f"Creating ChatAnthropic client: model={self.config.model_name}")
        
        return ChatAnthropic(
            model=self.config.model_name,
            temperature=self.config.temperature,
            anthropic_api_key=self.config.api_key,
            max_tokens=self.config.max_tokens or 4096,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    def _create_ollama_client(self) -> Any:
        """Create ChatOllama client for local models."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "langchain-ollama is not installed. Install with: pip install langchain-ollama"
            )
        
        self.logger.info(f"Creating ChatOllama client: model={self.config.model_name}")
        
        return ChatOllama(
            model=self.config.model_name,
            temperature=self.config.temperature,
            base_url=self.config.base_url or "http://localhost:11434",
        )

    def _convert_to_messages(
        self, messages: list[dict[str, Any] | BaseMessage]
    ) -> list[BaseMessage]:
        """
        Convert message dicts to LangChain message objects.

        Args:
            messages: List of message dicts or BaseMessage objects.

        Returns:
            List of LangChain BaseMessage objects.
        """
        result: list[BaseMessage] = []

        for msg in messages:
            if isinstance(msg, BaseMessage):
                result.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    result.append(SystemMessage(content=content))
                elif role == "assistant":
                    result.append(AIMessage(content=content))
                elif role == "tool":
                    # Claude/Bedrock requires content to be a list format
                    tool_content = content
                    if self._is_claude_model() and isinstance(content, str):
                        tool_content = [{"type": "text", "text": content}]
                    result.append(
                        ToolMessage(
                            content=tool_content,
                            tool_call_id=msg.get("tool_call_id", ""),
                        )
                    )
                else:  # user or default
                    result.append(HumanMessage(content=content))

        return result

    async def ainvoke(
        self,
        messages: list[dict[str, Any] | BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIMessage:
        """
        Asynchronously invoke the LLM with retry logic.

        Args:
            messages: List of messages (dicts or BaseMessage objects).
            tools: Optional list of tool schemas for function calling.
                   Should be in OpenAI function calling format.

        Returns:
            AIMessage containing the LLM response.

        Example:
            >>> response = await client.ainvoke(
            ...     messages=[{"role": "user", "content": "What is 2+2?"}],
            ...     tools=[{"type": "function", "function": {...}}]
            ... )
            >>> print(response.content)
        """
        self.total_calls += 1
        langchain_messages = self._convert_to_messages(messages)

        self.logger.debug(
            f"Invoking LLM [{self.config.provider}] with {len(langchain_messages)} messages, "
            f"tools={len(tools) if tools else 0}"
        )

        # Retry logic
        last_error = None
        delay = 1.0
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Bind tools if provided
                if tools:
                    llm_with_tools = self.llm.bind_tools(tools)
                    response = await llm_with_tools.ainvoke(langchain_messages)
                else:
                    response = await self.llm.ainvoke(langchain_messages)

                self.logger.debug(
                    f"LLM response received: content_length={len(response.content) if response.content else 0}, "
                    f"tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0}"
                )

                return response
                
            except Exception as e:
                last_error = e
                self.retry_count += 1
                
                if attempt < self.config.max_retries:
                    # Exponential backoff with jitter
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = min(delay + jitter, 60.0)
                    
                    self.logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay = min(delay * 2, 60.0)
                else:
                    self.failed_calls += 1
                    self.logger.error(f"LLM call failed after {self.config.max_retries + 1} attempts: {e}")
                    raise last_error

        return response

    async def ainvoke_simple(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> str:
        """
        Simple async invocation with just prompt and optional system message.

        Args:
            prompt: User prompt.
            system_message: Optional system message.

        Returns:
            Response content as string.
        """
        messages: list[dict[str, Any]] = []

        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = await self.ainvoke(messages)
        return response.content or ""

    async def astream(
        self,
        messages: list[dict[str, Any] | BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream LLM responses asynchronously.

        Args:
            messages: List of messages.
            tools: Optional list of tool schemas.

        Yields:
            Response chunks as strings.

        Example:
            >>> async for chunk in client.astream([{"role": "user", "content": "Tell me a story"}]):
            ...     print(chunk, end="", flush=True)
        """
        langchain_messages = self._convert_to_messages(messages)

        self.logger.debug(f"Starting LLM stream with {len(langchain_messages)} messages")

        if tools:
            llm_with_tools = self.llm.bind_tools(tools)
            async for chunk in llm_with_tools.astream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        else:
            async for chunk in self.llm.astream(langchain_messages):
                if chunk.content:
                    yield chunk.content

    def has_tool_calls(self, response: AIMessage) -> bool:
        """
        Check if the response contains tool calls.

        Args:
            response: AIMessage from the LLM.

        Returns:
            True if there are tool calls to process.
        """
        return bool(hasattr(response, "tool_calls") and response.tool_calls)

    def get_tool_calls(self, response: AIMessage) -> list[dict[str, Any]]:
        """
        Extract tool calls from an AIMessage.

        Args:
            response: AIMessage from the LLM.

        Returns:
            List of tool call dicts with 'name', 'args', and 'id'.
        """
        if not self.has_tool_calls(response):
            return []

        return [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }
            for tc in response.tool_calls
        ]

    def create_tool_message(
        self, tool_call_id: str, content: str
    ) -> ToolMessage:
        """
        Create a ToolMessage for feeding back to the LLM.

        Args:
            tool_call_id: ID of the tool call this responds to.
            content: Result content from the tool.

        Returns:
            ToolMessage instance.
        """
        # Claude/Bedrock requires content to be a list format
        if self._is_claude_model():
            tool_content = [{"type": "text", "text": content}]
        else:
            tool_content = content
        return ToolMessage(content=tool_content, tool_call_id=tool_call_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get LLM call statistics."""
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "retry_count": self.retry_count,
            "success_rate": (self.total_calls - self.failed_calls) / self.total_calls if self.total_calls > 0 else 0,
        }

    def print_statistics(self) -> None:
        """Print LLM call statistics."""
        stats = self.get_statistics()
        print("\n<LLM Service Statistics>")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model_name}")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Retry count: {stats['retry_count']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print("</LLM Service Statistics>\n")

    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"LLMClient("
            f"provider={self.config.provider}, "
            f"model={self.config.model_name})"
        )


# === Utility Functions ===

def get_available_providers() -> dict[str, bool]:
    """Check which LLM providers are available."""
    return {
        "openai": OPENAI_AVAILABLE,
        "anthropic": ANTHROPIC_AVAILABLE,
        "google": GOOGLE_AVAILABLE,
        "ollama": OLLAMA_AVAILABLE,
    }


def list_available_providers() -> list[str]:
    """List providers that have API keys configured."""
    providers = []
    if os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    if os.getenv("GOOGLE_API_KEY"):
        providers.append("google")
    if os.getenv("DEEPSEEK_API_KEY"):
        providers.append("deepseek")
    if os.getenv("QWEN_API_KEY"):
        providers.append("qwen")
    providers.append("ollama")  # Always available locally
    return providers


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    async def main():
        load_dotenv(override=True)
        
        print("=" * 60)
        print("LLM Client Test")
        print("=" * 60)
        
        # Show available providers
        print(f"\nInstalled SDKs: {get_available_providers()}")
        print(f"Configured providers: {list_available_providers()}")
        
        # Create client
        config = LLMConfig()
        print(f"\nConfig: provider={config.provider}, model={config.model_name}")
        
        client = LLMClient(config)
        print(f"Client: {client}")
        
        # Test simple invocation
        print("\n--- Test 1: Simple Invocation ---")
        response = await client.ainvoke_simple(
            prompt="Hello, how are you? Reply in one sentence.",
            system_message="You are a helpful assistant."
        )
        print(f"Response: {response}")
        
        # Test tool calling
        print("\n--- Test 2: Tool Calling ---")
        tools = [{
            "type": "function",
            "function": {
                "name": "search_papers",
                "description": "Search for academic papers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }]
        
        response = await client.ainvoke(
            messages=[{"role": "user", "content": "Search for papers about perovskite solar cells"}],
            tools=tools
        )
        
        tool_calls = client.get_tool_calls(response)
        print(f"Content: {response.content[:100] if response.content else 'None'}...")
        print(f"Tool calls: {tool_calls}")
        
        if tool_calls:
            print("✅ Tool calling works!")
        else:
            print("⚠️  No tool calls returned. Check if provider supports tool calling.")
        
        # Print statistics
        client.print_statistics()

    asyncio.run(main())