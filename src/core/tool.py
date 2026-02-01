"""
MCP Tool Registry for PSC_Agents.

Manages MCP server lifecycles: Connect -> Discover -> Execute.
This module serves as the "Hands" of the agent system.

Author: PSC_Agents Team
"""

import logging
import json
import base64
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Any
import httpx

# MCP SDK imports - wrapped for graceful degradation
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client  # <--- new add to approve sse
    from mcp.client.stdio import StdioServerParameters, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore

from core.config import MCPConfig, MCPServerConfig


def create_no_proxy_httpx_client(**kwargs):
    """Create an httpx AsyncClient without proxy to avoid SSE connection issues."""
    # 禁用代理，避免 SSE 长连接通过代理时出现超时问题
    return httpx.AsyncClient(proxy=None, **kwargs)


# === Local Tool Definitions ===
# NOTE: Agent-specific tools should be defined in their respective agent classes.
# - DataAgent tools: save_markdown_locally, extract_data_from_papers -> data_agent.py
# - DesignAgent tools: generate_material_structure, etc. -> design_agent.py
# This keeps the tool registry clean and agent-specific.

LOCAL_TOOLS: list[dict] = []  # Empty - agent-specific tools are in their own modules


class MCPToolRegistry:
    """
    Registry for MCP tools across multiple servers.

    Manages the lifecycle of MCP server connections and provides
    a unified interface for tool discovery and execution.

    Attributes:
        config: MCPConfig containing server definitions.
        sessions: Active ClientSession objects keyed by server name.
        tool_map: Mapping from tool_name to server_name for routing.
        tools_cache: Cached tool schemas for LLM consumption.
        local_save_dir: Directory for saving downloaded files locally.
        local_data_dir: Directory for saving local data files.

    Example:
        >>> config = MCPConfig.from_dict({
        ...     "arxiv": {"command": "uvx", "args": ["arxiv-mcp-server"]}
        ... })
        >>> registry = MCPToolRegistry(config, local_save_dir="./papers", local_data_dir="./data")
        >>> async with registry:
        ...     tools = await registry.get_tools_schema()
        ...     result = await registry.call_tool("search_papers", {"query": "perovskite"})
    """

    # Local tools that don't go through MCP
    # NOTE: Agent-specific tools are now handled in their own agent classes
    LOCAL_TOOL_NAMES: set[str] = set()

    def __init__(
        self, 
        config: MCPConfig,
        local_save_dir: str | Path = "../test/papers/",
        local_data_dir: str | Path = "../test/data/"
    ) -> None:
        """
        Initialize the tool registry.

        Args:
            config: MCPConfig with server definitions.
            local_save_dir: Directory for saving downloaded files locally.
            local_data_dir: Directory for saving local data files.
        """
        self.config = config
        self.local_save_dir = Path(local_save_dir)
        self.local_data_dir = Path(local_data_dir)
        self.sessions: dict[str, ClientSession] = {}
        self.tool_map: dict[str, str] = {}  # tool_name -> server_name
        self.tools_cache: list[dict[str, Any]] = []
        self._exit_stack = AsyncExitStack()
        self._initialized = False

        self.logger = logging.getLogger(f"{__name__}.MCPToolRegistry")

    async def initialize(self) -> dict[str, bool]:
        """
        Connect to all configured MCP servers.

        Iterates through enabled servers, establishes connections using
        StdioServerParameters, and stores active sessions.

        Returns:
            Dictionary mapping server names to connection success status.

        Raises:
            RuntimeError: If MCP SDK is not installed.
        """
        if not MCP_AVAILABLE:
            raise RuntimeError(
                "MCP SDK is not installed. Install with: pip install mcp"
            )

        if self._initialized:
            self.logger.warning("Registry already initialized")
            return {name: True for name in self.sessions}

        results: dict[str, bool] = {}
        enabled_servers = self.config.get_enabled_servers()

        for server_name, server_config in enabled_servers.items():
            success = await self._connect_server(server_name, server_config)
            results[server_name] = success

        self._initialized = True
        self.logger.info(
            f"Initialization complete: {sum(results.values())}/{len(results)} servers connected"
        )

        # Refresh tools after connecting
        await self.get_tools_schema()

        return results

    async def _connect_server(
        self, server_name: str, server_config: MCPServerConfig
    ) -> bool:
        try:
            self.logger.info(f"Connecting to MCP server '{server_name}'...")

            # 1. 判断连接类型：如果是 URL 则走 SSE，否则走 Stdio
            if hasattr(server_config, "url") and server_config.url:
                # === SSE 连接逻辑 ===
                self.logger.info(f"Using SSE transport for {server_name}")
                # sse_client 返回 (read_stream, write_stream)
                # 增加超时时间：timeout=30秒用于HTTP操作，sse_read_timeout=600秒用于SSE读取
                # 使用无代理的 httpx client 避免 SSE 长连接问题
                transport = await self._exit_stack.enter_async_context(
                    sse_client(
                        server_config.url, 
                        timeout=30.0, 
                        sse_read_timeout=600.0,
                        httpx_client_factory=create_no_proxy_httpx_client
                    )
                )
            elif server_config.command:
                # === Stdio 连接逻辑 (原有的代码) ===
                server_params = StdioServerParameters(
                    command=server_config.command,
                    args=server_config.args,
                    env=server_config.env,
                )
                transport = await self._exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
            else:
                raise ValueError(f"Server '{server_name}' must have either 'command' or 'url'")

            read_stream, write_stream = transport

            # 2. 创建并初始化会话 (这部分通用)
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            self.sessions[server_name] = session
            self.logger.info(f"Successfully connected to '{server_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to '{server_name}': {e}")
            return False
        

    async def shutdown(self) -> None:
        """
        Disconnect from all MCP servers.

        Closes the exit stack, which cleanly closes all managed contexts.
        """
        self.logger.info("Shutting down MCP connections...")
        await self._exit_stack.aclose()
        self.sessions.clear()
        self.tool_map.clear()
        self.tools_cache.clear()
        self._initialized = False
        self.logger.info("All MCP connections closed")

    async def get_tools_schema(self) -> list[dict[str, Any]]:
        """
        Fetch and aggregate tool schemas from all connected servers.

        Queries each active session for available tools, populates
        the tool_map for routing, and returns schemas formatted for LLM.
        Also includes local tools for file saving.

        Returns:
            List of tool schema dictionaries compatible with OpenAI function calling.
            Each dict contains: name, description, parameters (JSON schema).

        Example:
            >>> tools = await registry.get_tools_schema()
            >>> # Returns format compatible with LLM tool binding
            >>> # [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        """
        self.tool_map.clear()
        all_tools: list[dict[str, Any]] = []

        # Add local tools first
        for tool in LOCAL_TOOLS:
            all_tools.append(tool)
            tool_name = tool["function"]["name"]
            self.tool_map[tool_name] = "_local_"  # Special marker for local tools
        
        self.logger.debug(f"Added {len(LOCAL_TOOLS)} local tools")

        for server_name, session in self.sessions.items():
            try:
                self.logger.debug(f"Fetching tools from '{server_name}'...")
                response = await session.list_tools()

                for tool in response.tools:
                    # Build OpenAI-compatible function schema
                    tool_schema = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                        },
                    }
                    all_tools.append(tool_schema)

                    # Register for routing
                    self.tool_map[tool.name] = server_name

                self.logger.debug(
                    f"Fetched {len(response.tools)} tools from '{server_name}'"
                )

            except Exception as e:
                self.logger.error(f"Failed to fetch tools from '{server_name}': {e}")

        self.tools_cache = all_tools
        self.logger.info(
            f"Tool discovery complete: {len(all_tools)} tools ({len(LOCAL_TOOLS)} local + {len(all_tools) - len(LOCAL_TOOLS)} MCP) from {len(self.sessions)} servers"
        )
        return all_tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """
        Execute a tool call, routing to the appropriate MCP server or local handler.

        Args:
            name: Name of the tool to invoke.
            args: Dictionary of arguments to pass to the tool.

        Returns:
            The tool's response. For text responses, returns the text directly.
            For multiple content items, returns a list.

        Raises:
            ValueError: If the tool is not found in any connected server.
            RuntimeError: If the target server session is not active.

        Example:
            >>> result = await registry.call_tool(
            ...     "search_papers",
            ...     {"query": "perovskite solar cells", "max_results": 5}
            ... )
        """
        # Check if it's a local tool
        if name in self.LOCAL_TOOL_NAMES:
            return await self._handle_local_tool(name, args)

        # Look up target server
        server_name = self.tool_map.get(name)
        if not server_name:
            available = list(self.tool_map.keys())
            raise ValueError(
                f"Tool '{name}' not found. Available tools: {available}"
            )

        # Get session
        session = self.sessions.get(server_name)
        if not session:
            raise RuntimeError(
                f"Server '{server_name}' session not active. Call initialize() first."
            )

        self.logger.info(f"Calling tool '{name}' on server '{server_name}'")
        self.logger.info(f"Tool arguments: {args}")  # Changed to info level for visibility
        
        # Print to console for debugging
        print(f"\n🔧 [Tool Call] {name}")
        print(f"   Args: {args}")

        # Execute
        result = await session.call_tool(name, arguments=args)

        # Process response
        if result.content:
            if len(result.content) == 1 and hasattr(result.content[0], "text"):
                return result.content[0].text
            return [
                {"type": "text", "text": c.text} if hasattr(c, "text")
                else {"type": "blob", "data": getattr(c, "data", str(c))}
                for c in result.content
            ]

        return None

    def get_tool_names(self) -> list[str]:
        """Return list of all available tool names."""
        return list(self.tool_map.keys())

    def is_initialized(self) -> bool:
        """Check if the registry has been initialized."""
        return self._initialized

    async def __aenter__(self) -> "MCPToolRegistry":
        """Async context manager entry - initializes connections."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - shuts down connections."""
        await self.shutdown()

    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"MCPToolRegistry("
            f"servers={len(self.sessions)}, "
            f"tools={len(self.tool_map)}, "
            f"initialized={self._initialized})"
        )
    

if __name__ == "__main__":
    import asyncio

    async def main():
        print("🚀 Starting MCP Discovery Test (Listing Tools Only)...")
        # Example usage
        config_data = {
            "matablgpt": {
                "url": "https://seuyishu-mattablegpt.hf.space/sse",
                "command": None, # SSE 模式下不需要
                "args": []
            }
        }
        config = MCPConfig.from_dict(config_data)

        async with MCPToolRegistry(config) as registry:
            if not registry.is_initialized():
                print("Failed to initialize MCPToolRegistry.")
                return
            # 3. 获取并展示工具
            print("\n📡 Fetching tools from MaTableGPT...")
            try:
                tools = await registry.get_tools_schema()
                
                print(f"\n✅ Discovery Complete! Found {len(tools)} tools:\n")
                print("="*60)
                
                for i, tool in enumerate(tools, 1):
                    func_def = tool.get("function", {})
                    name = func_def.get("name", "Unknown")
                    desc = func_def.get("description", "No description")
                    params = func_def.get("parameters", {})

                    print(f"🛠️  Tool #{i}: {name}")
                    print(f"📝 Description: {desc}")
                    print(f"📋 Parameters Schema: {list(params.get('properties', {}).keys())}")
                    print("-" * 60)
            
            except Exception as e:
                print(f"❌ Error during tool discovery: {e}")

            #工具测试
            import json
            # === 步骤 1: 准备测试数据 (输入) ===
            # 一个简单的 HTML 表格
            sample_html = """
            <table border="1">
                <thead>
                    <tr>
                        <th>Catalyst Name</th>
                        <th>Overpotential (mV)</th>
                        <th>Tafel Slope (mV/dec)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>NiFe-LDH</td>
                        <td>230</td>
                        <td>45</td>
                    </tr>
                    <tr>
                        <td>Co3O4</td>
                        <td>310</td>
                        <td>60</td>
                    </tr>
                </tbody>
            </table>
            """

            # === 步骤 2: 调用转换工具 (Tool #2) ===
            print("\n1️⃣ Converting HTML to TSV...")
            try:
                tsv_result = await registry.call_tool(
                    "html_to_tsv_representation",
                    {
                        "html_table": sample_html,
                        "title": "Test Catalyst Performance",
                        "table_name": "test_table_001"
                    }
                )
                print(f"✅ Conversion Result Type: {type(tsv_result)}")
                # 打印出来看看结构，确认下一步该传什么
                print(f"📦 Payload snippet: {str(tsv_result)[:100]}...") 

            except Exception as e:
                print(f"❌ Conversion failed: {e}")
                return



    asyncio.run(main())