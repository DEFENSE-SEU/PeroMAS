"""
DataAgent - Literature Intelligence Officer for PSC_Agents.

Core Responsibilities:
1. Topic Expansion: Break down research themes into searchable sub-topics.
2. Strategic Retrieval: Search, screen, and acquire relevant papers.
3. Precision Extraction: Deep-read selected papers to extract parameters.
4. Structured Output: Compile findings into actionable data context.

Author: PSC_Agents Team
"""

import sys
import json
import re
import shutil
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent
from core.config import Settings


# ============================================================================
# MCP 服务器配置
# ============================================================================

# ArXiv MCP Server - 论文搜索、下载、阅读
ARXIV_MCP_SERVER_URL = "https://seuyishu-arxiv-mcp-server.hf.space/sse"

# MCP 服务器列表 (DataAgent 专用)
DATA_AGENT_MCP_SERVERS = {
    "arxiv": {
        "url": ARXIV_MCP_SERVER_URL,
        "description": "ArXiv paper search, download, and read",
        "tools": ["search_papers", "download_paper", "read_paper"]
    }
}


# ============================================================================
# DataAgent 专属工具定义 (本地工具)
# 这些工具仅供 DataAgent 使用
# ============================================================================

DATA_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_markdown_locally",
            "description": """Save paper Markdown to local filesystem.

After read_paper, use this to save the content locally.
Content is auto-retrieved from cache - just provide save_path with paper_id.

Args:
  save_path: Path like "./papers/2311.09695.md" (must include paper_id)
  content: Optional - will use cached data from read_paper
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Markdown content (optional - uses cache)"
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Local path with paper_id in filename (e.g., './papers/2311.09695.md')"
                    }
                },
                "required": ["save_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_data_from_papers",
            "description": """Extract structured data from all downloaded papers.

Reads all .md files in papers_dir and uses LLM to extract data based on goal/plan.
Call this AFTER all papers are downloaded and saved.

Returns: JSON with extracted data from each paper.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Research goal (what data to extract)"
                    },
                    "plan": {
                        "type": "string",
                        "description": "Specific parameters/metrics to extract"
                    },
                    "papers_dir": {
                        "type": "string",
                        "description": "Directory containing .md paper files"
                    }
                },
                "required": ["goal", "plan", "papers_dir"]
            }
        }
    }
]


# === System Prompt ===
SYSTEM_PROMPT = """You are DataAgent - Literature Intelligence Officer of PSC_Agents.

## 🎯 Core Mission
Retrieve and extract structured data from scientific literature to support perovskite solar cell research.
You have full autonomy to decide which tools to use and in what order based on the task requirements.

## 🛠️ Your Toolbox
| Tool | Purpose | Notes |
|------|---------|-------|
| `search_papers` | Search arXiv | Use simple keyword queries |
| `download_paper` | Download by ID | Returns paper metadata |
| `read_paper` | PDF → Markdown | Extracts full text |
| `save_markdown_locally` | Save to disk | Required before extraction |
| `extract_data_from_papers` | LLM-powered extraction | Only works on saved local files |

## 🔴 Technical Constraint: Search Query Format
arXiv API has limited query parsing. Complex Boolean queries will fail or return garbage.

**What works:**
- Simple keyword phrases: `CsPbI3 doping stability perovskite`
- Natural language: `perovskite solar cell efficiency`
- Domain-specific terms: `halide perovskite black phase`

**What fails:**
- Boolean operators: `(A OR B) AND (C OR D)` → ❌
- Nested parentheses: `ti:(X AND (Y OR Z))` → ❌
- Long OR lists: `(Mn OR Zn OR Ni OR Co OR Cd)` → ❌

**Tip:** If you need to cover multiple aspects, run multiple separate searches with different simple queries.

## 📌 Tool Dependency
`extract_data_from_papers` reads from the local papers directory. Papers must be saved locally first.
Typical sequence: download_paper → read_paper → save_markdown_locally → extract_data_from_papers

## 💡 Decision Guidelines
- **Need literature overview?** → Search with multiple simple queries
- **Need specific paper content?** → Download → Read
- **Need structured data extraction?** → Save papers locally first, then extract
- **Task says "extract" or "analyze"?** → You'll likely need to save papers before extracting

## Output Principles
- Only report information found in actual papers
- Never hallucinate references or data
- Provide structured output when extracting data
"""


class DataAgent(BaseAgent):
    """
    Data Agent: Search -> Screen -> Download -> Extract literature data.
    
    Local Tools (DataAgent-specific):
        - save_markdown_locally: 保存论文 Markdown 到本地
        - extract_data_from_papers: 从论文中提取结构化数据
    """
    
    # DataAgent 专属工具名称
    LOCAL_DATA_TOOLS = {"save_markdown_locally", "extract_data_from_papers"}

    def __init__(
        self, 
        settings: Settings | None = None,
        max_papers_per_topic: int = 5,
        local_papers_dir: str | None = None,
        local_data_dir: str | None = None,
    ) -> None:
        super().__init__(name="DataAgent", settings=settings)
        self.max_papers_per_topic = max_papers_per_topic
        # Default to test/papers relative to this file
        if local_papers_dir is None:
            self.local_papers_dir = str(Path(__file__).parent.parent / "test" / "papers")
        else:
            self.local_papers_dir = local_papers_dir
        if local_data_dir is None:
            self.local_data_dir = str(Path(__file__).parent.parent / "test" / "data")
        else:
            self.local_data_dir = local_data_dir
        self._markdown_cache: dict[str, str] = {}  # paper_id -> markdown content
        # 存储专属工具的 schema
        self._data_tools = DATA_AGENT_TOOLS.copy()

    def _get_system_prompt(
        self,
        state: dict[str, Any],
        default_prompt: str | None = None,
    ) -> str:
        return SYSTEM_PROMPT

    def _process_tool_output(self, output: str, tool_name: str) -> str | None:
        """
        Override: 
        - Print search results for visibility.
        - Apply Topic Guard filter to remove irrelevant papers.
        - Cache Markdown content from read_paper, return metadata only.
        """
        # Print arXiv search results with Topic Guard filtering
        if tool_name == "search_papers":
            try:
                data = json.loads(output)
                papers = data.get("papers", [])
                total = data.get("total_results", len(papers))
                
                # === Topic Guard Filter (Enhanced) ===
                # MUST have at least one core keyword
                CORE_KEYWORDS = {
                    "perovskite", "halide", "photovoltaic", "pv cell",
                    "mapbi", "fapbi", "cspbi", "fasni", "mapi", "fapi"
                }
                # Supporting keywords (need core + supporting)
                SUPPORTING_KEYWORDS = {
                    "solar", "efficiency", "pce", "device", "cell", 
                    "stability", "degradation", "film", "absorber", 
                    "transport", "electron", "hole", "band gap",
                    "methylammonium", "formamidinium", "cesium", "lead", "tin",
                    "iodide", "bromide", "chloride", "fabrication", "annealing",
                    "spin-coating", "vapor deposition", "etl", "htl"
                }
                # Definitely NOT relevant (astrophysics, math, etc.)
                IRRELEVANT_KEYWORDS = {
                    "astrophysics", "cosmology", "magnetic field", "topology", "lipschitz",
                    "heliosphere", "gravitational", "neutrino", "quark",
                    "galaxy", "stellar", "plasma physics", "fusion", "superconductor",
                    "dark matter", "black hole", "quantum computing", "cryptography",
                    "solar wind", "solar orbiter", "solar flare", "coronal", 
                    "magnetosphere", "interplanetary", "spacecraft", "telescope"
                }
                
                relevant_papers = []
                skipped_papers = []
                
                for paper in papers:
                    title_lower = paper.get("title", "").lower()
                    abstract_lower = paper.get("abstract", "")[:500].lower() if paper.get("abstract") else ""
                    combined = title_lower + " " + abstract_lower
                    
                    # Check for irrelevant keywords first (instant reject)
                    is_irrelevant = any(kw in combined for kw in IRRELEVANT_KEYWORDS)
                    
                    # Must have at least one CORE keyword
                    has_core = any(kw in combined for kw in CORE_KEYWORDS)
                    
                    # Should also have supporting context
                    has_supporting = any(kw in combined for kw in SUPPORTING_KEYWORDS)
                    
                    # Accept if: has core keyword, not irrelevant
                    # Also accept if: has "solar cell" AND supporting keywords (but not astrophysics)
                    is_solar_cell_paper = "solar cell" in combined and has_supporting
                    
                    if is_irrelevant:
                        skipped_papers.append(paper)
                    elif has_core or is_solar_cell_paper:
                        relevant_papers.append(paper)
                    else:
                        skipped_papers.append(paper)
                
                # Print results
                print(f"\n{'─'*60}")
                print(f"🔍 [arXiv] {total} papers found, {len(relevant_papers)} relevant (Topic Guard filtered)")
                print(f"{'─'*60}")
                
                if relevant_papers:
                    print("  ✅ RELEVANT (will download):")
                    for i, paper in enumerate(relevant_papers[:8], 1):
                        arxiv_id = paper.get("id", "N/A")
                        title = paper.get("title", "N/A")[:70]
                        print(f"    [{i}] {arxiv_id}: {title}...")
                
                if skipped_papers:
                    print(f"\n  ❌ SKIPPED ({len(skipped_papers)} irrelevant):")
                    for paper in skipped_papers[:3]:
                        title = paper.get("title", "N/A")[:50]
                        print(f"    - {title}...")
                    if len(skipped_papers) > 3:
                        print(f"    ... and {len(skipped_papers) - 3} more")
                
                print(f"{'─'*60}\n")
                
                # Return filtered results
                if relevant_papers:
                    filtered_data = {
                        "papers": relevant_papers,
                        "total_results": len(relevant_papers),
                        "original_total": total,
                        "filtered_count": len(skipped_papers),
                        "filter_note": "Topic Guard applied - only perovskite/solar related papers"
                    }
                    return json.dumps(filtered_data, indent=2)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse search results: {e}")
            return None
        
        # Cache Markdown content from read_paper
        if tool_name == "read_paper" and "content" in output:
            try:
                data = json.loads(output)
                if data.get("status") == "success" and "content" in data:
                    paper_id = data.get("paper_id", "unknown")
                    self._markdown_cache[paper_id] = data["content"]
                    
                    # Return metadata only (content is too large for LLM context)
                    content_preview = data["content"][:500] + "..." if len(data["content"]) > 500 else data["content"]
                    metadata = {
                        "status": "success",
                        "paper_id": paper_id,
                        "title": data.get("title"),
                        "content_length": len(data["content"]),
                        "content_preview": content_preview,
                        "message": f"Markdown cached. Call save_markdown_locally with save_path containing '{paper_id}.md'"
                    }
                    self.logger.info(f"Cached Markdown for {paper_id} ({len(data['content'])} chars)")
                    return json.dumps(metadata, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to cache Markdown: {e}")
        return None

    def _preprocess_tool_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Override: Inject cached Markdown content for save_markdown_locally.
        """
        if tool_name == "save_markdown_locally":
            save_path = args.get("save_path", "")
            # Try arXiv ID pattern (e.g., 2401.12345 or 2401.12345v1)
            match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', save_path)
            if match:
                paper_id = match.group(1)
                if paper_id in self._markdown_cache:
                    self.logger.info(f"Injecting cached Markdown for arXiv {paper_id}")
                    return {"content": self._markdown_cache[paper_id], "save_path": save_path}
        return args

    # =========================================================================
    # DataAgent 专属工具处理
    # =========================================================================
    
    async def _get_tools_with_data_tools(self) -> list[dict[str, Any]]:
        """获取工具列表，包含 DataAgent 专属工具"""
        # 从 registry 获取 MCP 工具
        mcp_tools = []
        if self.registry.is_initialized():
            mcp_tools = await self.registry.get_tools_schema()
        
        # 合并专属数据工具
        all_tools = self._data_tools + mcp_tools
        return all_tools

    async def _handle_data_tool(self, name: str, args: dict[str, Any]) -> str:
        """处理 DataAgent 专属工具调用"""
        if name == "save_markdown_locally":
            return self._save_markdown_locally(args)
        elif name == "extract_data_from_papers":
            return await self._extract_data_from_papers(args)
        else:
            return json.dumps({"error": f"Unknown data tool: {name}"})

    def _save_markdown_locally(self, args: dict[str, Any]) -> str:
        """Save Markdown content to local filesystem."""
        try:
            content = args.get("content", "")
            save_path = args.get("save_path", "")
            
            if not save_path:
                return json.dumps({"status": "error", "message": "save_path is required"})
            
            # Check if content is missing
            if not content:
                return json.dumps({
                    "status": "error", 
                    "message": "content is missing. Make sure read_paper was called first and the paper_id is in the save_path."
                })
            
            # Ensure directory exists
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save Markdown content
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            file_size = len(content.encode("utf-8"))
            self.logger.info(f"Markdown saved to {save_path} ({file_size} bytes)")
            
            return json.dumps({
                "status": "success",
                "message": f"Markdown saved successfully to {save_path}",
                "file_path": str(save_path.absolute()),
                "file_size": file_size,
                "file_size_kb": round(file_size / 1024, 2)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to save Markdown: {e}")
            return json.dumps({"status": "error", "message": f"Failed to save Markdown: {str(e)}"})

    async def _extract_data_from_papers(self, args: dict[str, Any]) -> str:
        """Extract structured data from all Markdown papers using LLM."""
        try:
            goal = args.get("goal", "")
            plan = args.get("plan", "")
            papers_dir = args.get("papers_dir", "")
            
            if not papers_dir:
                return json.dumps({"status": "error", "message": "papers_dir is required"})
            
            papers_path = Path(papers_dir)
            if not papers_path.exists():
                return json.dumps({"status": "error", "message": f"Directory not found: {papers_dir}"})
            
            # Find all .md files
            md_files = list(papers_path.glob("*.md"))
            if not md_files:
                return json.dumps({"status": "error", "message": f"No .md files found in {papers_dir}"})
            
            print(f"\n{'='*60}")
            print(f"📖 [Extract] Starting data extraction from {len(md_files)} papers...")
            print(f"{'='*60}")
            
            # Use self.llm if available
            if not self.llm:
                return json.dumps({"status": "error", "message": "LLM not configured"})
            
            extracted_data = []
            
            for i, md_file in enumerate(md_files, 1):
                paper_id = md_file.stem
                print(f"\n[{i}/{len(md_files)}] 📄 Processing: {paper_id}")
                
                try:
                    content = md_file.read_text(encoding="utf-8")
                    
                    # Truncate if too long
                    max_content_len = 15000
                    if len(content) > max_content_len:
                        content = content[:max_content_len] + "\n\n[... content truncated ...]"
                    
                    extraction_prompt = self._build_extraction_prompt(goal, plan, paper_id, content)
                    
                    messages = [
                        {"role": "system", "content": "You are a scientific data extraction expert. Extract structured data from research papers accurately. Output ONLY valid JSON, no other text."},
                        {"role": "user", "content": extraction_prompt}
                    ]
                    
                    response = await self.llm.ainvoke(messages)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Handle None response
                    if response_text is None:
                        raise ValueError("LLM returned None response")
                    
                    paper_data = self._parse_extraction_response(response_text, paper_id)
                    extracted_data.append(paper_data)
                    
                    # Safely get title (handle both missing key and None value)
                    title = paper_data.get('title') or 'Unknown'
                    print(f"    ✅ Extracted: {title[:50]}...")
                    
                except Exception as e:
                    self.logger.error(f"Failed to extract from {paper_id}: {e}")
                    extracted_data.append({
                        "arxiv_id": paper_id,
                        "status": "error",
                        "error": str(e)
                    })
                    print(f"    ❌ Error: {e}")
            
            print(f"\n{'='*60}")
            print(f"✅ Extraction complete: {len(extracted_data)} papers processed")
            print(f"{'='*60}\n")
            
            return json.dumps({
                "status": "success",
                "total_papers": len(extracted_data),
                "papers_dir": str(papers_dir),
                "extracted_data": extracted_data
            }, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Failed to extract data from papers: {e}")
            return json.dumps({"status": "error", "message": f"Extraction failed: {str(e)}"})

    def _build_extraction_prompt(self, goal: str, plan: str, paper_id: str, content: str) -> str:
        """Build the prompt for LLM data extraction."""
        return f"""# Task: Extract Data from Research Paper

## Research Goal
{goal}

## Research Plan (What to Extract)
{plan}

## Paper ID
{paper_id}

## Paper Content
{content}

## Instructions
Based on the Goal and Plan above, extract relevant data from this paper.

**Output Format (JSON only):**
```json
{{
  "paper_id": "{paper_id}",
  "title": "<paper title>",
  "year": <publication year as int>,
  "authors": ["<author1>", "<author2>", ...],
  "relevance": "<brief explanation of why this paper is relevant>",
  "key_findings": {{
    "<parameter_from_plan>": "<value with units>",
    "<another_parameter>": "<value with units>"
  }},
  "performance_metrics": {{
    "PCE": <float or null>,
    "Voc": <float or null>,
    "Jsc": <float or null>,
    "FF": <float or null>
  }},
  "materials": {{
    "composition": "<perovskite composition>",
    "additives": ["<additive1>", ...],
    "processing": "<fabrication method>"
  }},
  "notes": "<any important observations>"
}}
```

**Rules:**
1. Only extract values explicitly stated in the paper
2. Use null for values not found (never invent data)
3. Include units where applicable
4. Extract ALL performance metrics if available
5. Output ONLY the JSON, no other text
"""

    def _parse_extraction_response(self, response_text: str, paper_id: str) -> dict[str, Any]:
        """Parse LLM response to extract JSON data."""
        try:
            # Handle None or empty response
            if not response_text:
                return {
                    "arxiv_id": paper_id,
                    "status": "error",
                    "error": "Empty or None response from LLM"
                }
            
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "{" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text
            
            data = json.loads(json_str)
            data["arxiv_id"] = paper_id
            return data
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON for {paper_id}: {e}")
            return {
                "arxiv_id": paper_id,
                "status": "parse_error",
                "raw_response": response_text[:500]
            }

    async def autonomous_thinking(
        self,
        prompt: str,
        state: dict[str, Any],
        system_message: str | None = None,
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """重写 autonomous_thinking 以支持 DataAgent 专属工具"""
        if not self.llm:
            self.logger.error("LLM client not available")
            return {
                "response": "[ERROR] LLM not configured",
                "tool_calls": [],
                "tool_results": [],
                "iterations": 0,
            }

        # 获取工具 (包含专属数据工具)
        tools = await self._get_tools_with_data_tools()
        
        self.logger.info(f"Available tools: {len(tools)}")
        tool_names = [t.get('function', {}).get('name', 'unknown') for t in tools]
        self.logger.info(f"Tool names: {tool_names}")

        # Build messages
        messages: list[dict[str, Any]] = []
        
        final_system_prompt = self._get_system_prompt(state, system_message)
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})

        context_str = ""
        if state:
            context_str = f"\n\nCurrent context:\n{state}"

        messages.append({"role": "user", "content": prompt + context_str})

        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []
        iterations = 0
        response = None
        
        # Track consecutive tool calls for deduplication
        _last_tool_name: str | None = None
        _consecutive_count: int = 0

        # ReAct loop
        while iterations < max_iterations:
            iterations += 1
            self.logger.debug(f"Thinking iteration {iterations}")

            response = await self.llm.ainvoke(messages, tools=tools if tools else None)

            if not self.llm.has_tool_calls(response):
                self.logger.debug("No tool calls, finishing")
                break

            tool_calls = self.llm.get_tool_calls(response)
            messages.append(response)

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                self.logger.info(f"Executing tool: {tool_name}")
                
                # === 工具调用可视化 (去重逻辑) ===
                tool_type = "📍 Local" if tool_name in self.LOCAL_DATA_TOOLS else "🌐 MCP"
                if tool_name == _last_tool_name:
                    _consecutive_count += 1
                    print(f"\r   🔄 [DataAgent] {tool_name} called {_consecutive_count}x (consecutive)", end="", flush=True)
                else:
                    if _last_tool_name is not None and _consecutive_count > 1:
                        print()  # 换行
                    _consecutive_count = 1
                    _last_tool_name = tool_name
                    print(f"\n🔧 [DataAgent] Calling {tool_type} Tool: {tool_name}")
                    print(f"   📥 Arguments: {str(tool_args)[:200]}{'...' if len(str(tool_args)) > 200 else ''}")
                
                all_tool_calls.append(tc)

                # 预处理参数 (注入缓存内容)
                tool_args = self._preprocess_tool_args(tool_name, tool_args)

                try:
                    # 判断是专属工具还是 MCP 工具
                    if tool_name in self.LOCAL_DATA_TOOLS:
                        result_str = await self._handle_data_tool(tool_name, tool_args)
                    else:
                        result = await self.registry.call_tool(tool_name, tool_args)
                        result_str = str(result) if result else "No result"
                    
                    # 处理工具输出 (缓存、截断等)
                    processed = self._process_tool_output(result_str, tool_name)
                    if processed is not None:
                        result_str = processed
                    else:
                        result_str = self._truncate_tool_output(result_str, tool_name)
                        
                except Exception as e:
                    self.logger.error(f"Tool execution failed: {e}")
                    result_str = f"[ERROR] {e}"

                all_tool_results.append({
                    "tool": tool_name,
                    "result": result_str,
                })
                
                # === 工具结果可视化 (仅首次调用显示详细结果) ===
                if _consecutive_count == 1:
                    result_preview = result_str[:150] if len(result_str) > 150 else result_str
                    print(f"   📤 Result: {result_preview}{'...' if len(result_str) > 150 else ''}")

                tool_message = self.llm.create_tool_message(tool_id, result_str)
                messages.append(tool_message)

        final_response = ""
        if response and hasattr(response, "content"):
            final_response = response.content or ""

        return {
            "response": final_response,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "iterations": iterations,
        }

    def clear_papers(self) -> None:
        """
        Clear all papers in the papers directory.
        
        Call this method when the workflow iteration is FINISHED (not during iteration).
        This ensures papers are preserved during multi-round DataAgent calls within one workflow.
        
        Usage in workflow:
            # After workflow completes successfully
            data_agent.clear_papers()
        """
        papers_dir = Path(self.local_papers_dir)
        
        if papers_dir.exists():
            # Count files before cleaning
            md_files = list(papers_dir.glob("*.md"))
            if md_files:
                print(f"\n🗑️  [DataAgent] Clearing {len(md_files)} Markdown files from {papers_dir}")
                for md in md_files:
                    try:
                        md.unlink()
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {md}: {e}")
                print(f"✅ Papers directory cleared")
            else:
                print(f"📁 Papers directory is already empty: {papers_dir}")
    
    def _ensure_papers_directory(self) -> None:
        """Ensure papers directory exists."""
        papers_dir = Path(self.local_papers_dir)
        if not papers_dir.exists():
            papers_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created papers directory: {papers_dir}")

    def _build_research_prompt(self, goal: str, plan: str) -> str:
        """Build concise research prompt."""
        my_task = self._extract_my_task(plan, "DataAgent")
        
        return f"""# Research Task
**Goal**: {goal}
**Plan**: {plan}

# Workflow
1. **Search**: Generate queries and use `search_papers` to find relevant papers (target: 15+)
2. **Download**: Use `download_paper` → `read_paper` → `save_markdown_locally` for each paper (target: ≥10 saved)
3. **Extract**: Call `extract_data_from_papers(goal="{goal}", plan="{plan}", papers_dir="{self.local_papers_dir}")`

# Required Output (JSON) → feeds into state["data_context"]
```json
{{
  "goal": "<string: research goal>",
  "papers_analyzed": <integer: number of papers processed>,
  "extracted_data": [
    {{
      "paper_id": "<string>",
      "title": "<string>",
      "key_findings": ["<string>", ...],
      "data_points": {{"materials": [...], "methods": [...], "performance": [...]}}
    }}
  ],
  "synthesis": "<string: overall summary relevant to goal>"
}}
```
"""
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute literature review based on MetaAgent's task."""
        print(f"\n{'='*60}")
        print(f"📚 [DataAgent] Literature Review")
        print(f"{'='*60}")
        
        # Show available tools with categorization
        tools = await self._get_tools_with_data_tools()
        local_tool_names = list(self.LOCAL_DATA_TOOLS)
        mcp_tool_names = []
        for t in tools:
            name = t.get('function', {}).get('name', 'unknown')
            if name not in self.LOCAL_DATA_TOOLS:
                mcp_tool_names.append(name)
        
        print(f"\n🛠️  Available Tools Summary:")
        print(f"   📍 Local Tools ({len(local_tool_names)}): {local_tool_names}")
        print(f"   🌐 MCP Tools ({len(mcp_tool_names)}): {mcp_tool_names}")
        print(f"   📊 Total: {len(tools)} tools")
        
        # Ensure papers directory exists
        self._ensure_papers_directory()
        
        # Show existing papers count
        papers_dir = Path(self.local_papers_dir)
        existing_papers = list(papers_dir.glob("*.md"))
        if existing_papers:
            print(f"📂 {len(existing_papers)} existing papers in directory")
        
        # Clear Markdown cache
        self._markdown_cache.clear()
        
        # Get context from state (DataAgent is first, so only has goal and plan)
        goal = state.get("goal", "")
        plan = state.get("plan", "")
        
        # Extract DataAgent-specific task from MetaAgent's plan
        my_task = self._extract_my_task(plan, "DataAgent")
        
        # === Display upstream context ===
        print(f"\n📊 Upstream Context:")
        print(f"   ├─ 🎯 Goal: {goal[:80]}{'...' if len(goal) > 80 else ''}")
        print(f"   ├─ 📝 Task: {my_task}")
        print(f"   └─ 📁 Output Dir: {self.local_papers_dir}")

        # Build prompt with specific task
        prompt = f"""# 🧠 Data Intelligence Officer

## Your Task
{my_task}

## Research Goal
{goal}

## Workspace
- **Papers Directory**: `{self.local_papers_dir}`
- **Existing Papers**: {len(existing_papers)} .md files already saved

## Available Actions
You have full autonomy to decide what to do. Consider:

1. **Search** (`search_papers`) - Find relevant papers on arXiv
   - ⚠️ Use simple keyword queries (3-5 words), NOT complex Boolean
   - Example: `CsPbI3 doping stability perovskite` ✅
   - NOT: `(CsPbI3 OR cesium) AND (Mn OR Zn)` ❌

2. **Download & Read** (`download_paper`, `read_paper`) - Get paper content

3. **Save** (`save_markdown_locally`) - Save papers to local directory

4. **Extract** (`extract_data_from_papers`) - LLM-powered structured extraction
   - ⚠️ Only works on papers saved in `{self.local_papers_dir}`
   - Call with: `extract_data_from_papers(goal="{goal[:100]}...", plan="...", papers_dir="{self.local_papers_dir}")`

## Think About
- What information does the task need?
- Are there existing papers I can use, or do I need to search?
- If I need to extract structured data, have I saved the papers locally first?

## Output
Provide your findings in a structured format. If extracting data, include:
- Papers analyzed
- Key findings per paper
- Synthesis relevant to the research goal
"""

        result = await self.autonomous_thinking(
            prompt=prompt,
            state=state,
            system_message=SYSTEM_PROMPT,
            max_iterations=80, 
        )

        response_text = result.get("response", "")
        tool_results = result.get("tool_results", [])

        # Display summary
        print(f"\n{'─'*60}")
        print(f"✅ [DataAgent] Literature Review Complete")
        print(f"   └─ Tool calls: {len(tool_results)}")

        return {
            "data_context": self._build_data_context(response_text, tool_results)
        }
    
    def _extract_my_task(self, plan: str | dict, agent_name: str) -> str:
        """Extract specific task for this agent from MetaAgent's plan."""
        if not plan:
            return "Search and extract relevant literature data"
        
        # If plan is already a dict, use it directly
        if isinstance(plan, dict):
            plan_data = plan
        else:
            # Try to parse JSON from string plan
            try:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', plan)
                if match:
                    plan_data = json.loads(match.group(1))
                elif '{' in plan:
                    start = plan.find('{')
                    end = plan.rfind('}') + 1
                    plan_data = json.loads(plan[start:end])
                else:
                    return plan  # Return whole plan if not JSON
            except (json.JSONDecodeError, KeyError):
                # Fallback: search for agent name in plan text
                if agent_name in str(plan):
                    lines = str(plan).split('\n')
                    for line in lines:
                        if agent_name in line:
                            return line.replace(agent_name, "").strip(': -')
                return str(plan)[:500]
        
        # Extract agent-specific task from plan_data
        agent_tasks = plan_data.get("agent_tasks", {})
        task = agent_tasks.get(agent_name, "")
        
        if task and str(task).upper() != "SKIP":
            return task
        elif str(task).upper() == "SKIP":
            return "SKIP"
        else:
            return plan_data.get("iteration_focus", "Search relevant literature")

    def _build_data_context(self, response: str, tool_results: list[dict[str, Any]]) -> str:
        """Build final JSON context for downstream agents."""
        
        # First, check tool_results for extract_data_from_papers result
        for result in tool_results:
            tool_name = result.get("tool_name", "")
            tool_result = result.get("result", "")
            
            if tool_name == "extract_data_from_papers":
                # Parse the extraction result
                try:
                    if isinstance(tool_result, str):
                        extraction_data = json.loads(tool_result)
                    else:
                        extraction_data = tool_result
                    
                    if extraction_data.get("status") == "success":
                        # Build context from extraction result
                        context = {
                            "papers_analyzed": extraction_data.get("total_papers", 0),
                            "extracted_data": extraction_data.get("extracted_data", []),
                            "papers_dir": extraction_data.get("papers_dir", ""),
                            "synthesis": response[:1000] if response else "Data extracted successfully"
                        }
                        return json.dumps(context, indent=2, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"⚠️ Failed to parse extract_data_from_papers result: {e}")
        
        # Fallback: Try to extract JSON from response text
        json_data = self._extract_json_block(response)
        
        if json_data:
            # Valid JSON found, return as string
            return json.dumps(json_data, indent=2, ensure_ascii=False)
        
        # Fallback: build basic JSON structure
        fallback = {
            "papers_analyzed": 0,
            "extracted_data": [],
            "synthesis": response[:1000] if response else "No data extracted"
        }
        return json.dumps(fallback, indent=2, ensure_ascii=False)

    def _extract_json_block(self, text: str) -> Any | None:
        """Extract JSON from text."""
        try:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return json.loads(match.group(1))
            match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
        return None