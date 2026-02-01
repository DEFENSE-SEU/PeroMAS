"""
Configuration Management for PSC_Agents.

Central source of truth for all settings including LLM configuration,
MCP server definitions, and project paths.

Supports multiple LLM providers with per-provider API keys and base URLs:
- OpenAI (GPT series) - supports proxy
- Anthropic (Claude series) - supports proxy
- Google (Gemini series) - supports proxy
- DeepSeek - supports proxy
- Qwen - supports proxy
- Ollama (local)

Author: PSC_Agents Team
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Auto-load .env from project root at module import time
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env", override=True)


# === Provider Configuration Dataclass ===

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    
    def is_valid(self) -> bool:
        """Check if provider has required configuration."""
        return bool(self.api_key and self.model)
    
    def uses_proxy(self) -> bool:
        """Check if this provider uses a proxy (non-native) API."""
        if not self.base_url:
            return False
        native_urls = [
            "api.openai.com",
            "api.anthropic.com", 
            "generativelanguage.googleapis.com",
            "api.deepseek.com",
            "dashscope.aliyuncs.com",
            "localhost",
            "127.0.0.1",
        ]
        return not any(native in self.base_url for native in native_urls)


@dataclass
class LLMConfig:
    """
    Configuration for the LLM client.

    Supports multiple providers with per-provider API keys and base URLs.
    When a provider uses a proxy base_url, it will use ChatOpenAI for compatibility.

    Attributes:
        provider: Selected LLM provider (openai, anthropic, google, deepseek, qwen, ollama).
        temperature: Sampling temperature (0.0 - 2.0).
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries on failure.
        
        # Per-provider configurations
        openai: OpenAI provider config
        anthropic: Anthropic provider config  
        google: Google provider config
        deepseek: DeepSeek provider config
        qwen: Qwen provider config
        ollama: Ollama provider config
    """

    # Global settings
    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    max_tokens: int | None = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "0")) or None
    )
    timeout: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "120"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3"))
    )
    
    # Per-provider configurations
    openai: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    ))
    
    anthropic: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    ))
    
    google: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        api_key=os.getenv("GOOGLE_API_KEY", ""),
        base_url=os.getenv("GOOGLE_BASE_URL", ""),
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    ))
    
    deepseek: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    ))
    
    qwen: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        api_key=os.getenv("QWEN_API_KEY", ""),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=os.getenv("QWEN_MODEL", "qwen-max"),
    ))
    
    ollama: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        api_key="",  # Ollama doesn't need API key
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3"),
    ))

    def get_current_provider_config(self) -> ProviderConfig:
        """Get the configuration for the currently selected provider."""
        provider_map = {
            "openai": self.openai,
            "anthropic": self.anthropic,
            "google": self.google,
            "deepseek": self.deepseek,
            "qwen": self.qwen,
            "ollama": self.ollama,
        }
        return provider_map.get(self.provider.lower(), self.openai)
    
    @property
    def api_key(self) -> str:
        """Get API key for current provider."""
        return self.get_current_provider_config().api_key
    
    @property
    def base_url(self) -> str:
        """Get base URL for current provider."""
        return self.get_current_provider_config().base_url
    
    @property
    def model_name(self) -> str:
        """Get model name for current provider."""
        return self.get_current_provider_config().model
    
    def uses_proxy(self) -> bool:
        """Check if current provider uses a proxy API."""
        return self.get_current_provider_config().uses_proxy()

    def is_valid(self) -> bool:
        """Check if the configuration has required fields."""
        if self.provider == "ollama":
            return bool(self.model_name)
        return bool(self.api_key and self.model_name)


@dataclass
class MCPServerConfig:
    """
    Configuration for a single MCP server.

    Attributes:
        command: Executable command (e.g., 'python', 'npx', 'uvx').
        args: Command line arguments.
        env: Optional environment variables.
        enabled: Whether this server is enabled.
    """
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    enabled: bool = True
    url: Optional[str] = None  # <--- 必须添加这一行！


@dataclass
class MCPConfig:
    """
    Configuration for all MCP servers.

    Attributes:
        servers: Dictionary mapping server names to their configurations.
    """

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, dict[str, Any]]) -> "MCPConfig":
        """
        Create MCPConfig from a dictionary.

        Args:
            config_dict: Dictionary mapping server names to config dicts.
                        Each config dict should have 'command', 'args', etc.

        Returns:
            MCPConfig instance.

        Example:
            >>> config = MCPConfig.from_dict({
            ...     "arxiv": {
            ...         "command": "uvx",
            ...         "args": ["arxiv-mcp-server"]
            ...     },
            ...     "filesystem": {
            ...         "command": "npx",
            ...         "args": ["-y", "@anthropic/mcp-server-filesystem", "/path"]
            ...     }
            ... })
        """
        servers = {}
        for name, cfg in config_dict.items():
            servers[name] = MCPServerConfig(
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env"),
                enabled=cfg.get("enabled", True),
                url=cfg.get("url"),  # 支持 SSE 连接
            )
        return cls(servers=servers)

    def get_enabled_servers(self) -> dict[str, MCPServerConfig]:
        """Return only enabled server configurations."""
        return {
            name: cfg for name, cfg in self.servers.items() if cfg.enabled
        }


@dataclass
class ProjectConfig:
    """
    Project-level configuration.

    Attributes:
        project_root: Root directory of the project.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_format: Logging format string.
    """

    project_root: str = field(
        default_factory=lambda: os.getenv(
            "PSC_PROJECT_ROOT",
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("PSC_LOG_LEVEL", "INFO")
    )
    log_format: str = field(
        default_factory=lambda: os.getenv(
            "PSC_LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )


@dataclass
class Settings:
    """
    Aggregated settings for the entire application.

    This is the main configuration class that combines all config sections.

    Example:
        >>> settings = Settings(
        ...     llm=LLMConfig(api_key="sk-xxx", base_url="https://proxy.com/v1"),
        ...     mcp=MCPConfig.from_dict({"arxiv": {"command": "uvx", "args": ["arxiv-mcp-server"]}}),
        ... )
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)


# Default settings instance (can be overridden)
def get_default_settings() -> Settings:
    """
    Get default settings from environment variables.

    Returns:
        Settings instance with values from environment.
    """
    return Settings()
