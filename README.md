# PSC_Agents: Multi-Agent Perovskite Solar Cell Research System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)

## 📖 Overview

PSC_Agents is an autonomous multi-agent system for perovskite solar cell (PSC) research. It orchestrates a team of specialized AI agents to automate the research workflow: from literature review → material design → performance prediction → analysis → knowledge archival.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PSC_Agents Workflow                               │
│                                                                         │
│   User Goal ──► MetaAgent ──► DataAgent ──► DesignAgent ──► FabAgent   │
│                    │              │              │              │       │
│                    ▼              ▼              ▼              ▼       │
│                  plan       data_context   exp_params     fab_results   │
│                    │              │              │              │       │
│                    └──────────────┴──────────────┴──────────────┘       │
│                                      │                                  │
│                                      ▼                                  │
│                               AnalysisAgent                             │
│                                      │                                  │
│                                      ▼                                  │
│                              analysis_report                            │
│                                      │                                  │
│                                      ▼                                  │
│                               MemoryAgent ──► memory_log                │
│                                      │                                  │
│                                      ▼                                  │
│                            ┌─────────────────┐                          │
│                            │  Next Iteration │───► Loop back to Meta    │
│                            └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture

### Agent Roles

| Agent | Role | Tools | Output |
|-------|------|-------|--------|
| **MetaAgent** | Chief Scientist - Strategic planning & orchestration | None (Pure reasoning) | `plan` |
| **DataAgent** | Literature Intelligence - Paper search & extraction | MCP: arXiv server | `data_context` |
| **DesignAgent** | Design Expert - Material & synthesis design | Server: MatterGen, CSLLM | `experimental_params` |
| **FabAgent** | Fabrication Engineer - Performance prediction | Local: RF models | `fab_results` |
| **AnalysisAgent** | Lead Analyst - Gap analysis & diagnosis | Local: Chemistry, SHAP | `analysis_report` |
| **MemoryAgent** | Knowledge Keeper - Archival & learning | None (Pure reasoning) | `memory_log` |

### State Flow

Each agent reads from upstream agents and writes to specific state fields:

```python
AgentState = {
    "goal": str,                 # User's research objective
    "plan": dict,                # MetaAgent's strategy
    "data_context": str,         # DataAgent's literature findings
    "experimental_params": dict, # DesignAgent's material recipe
    "fab_results": dict,         # FabAgent's predictions
    "analysis_report": str,      # AnalysisAgent's diagnosis
    "memory_log": list[str],     # All iteration knowledge capsules
    "current_iteration": int,    # Loop counter
    "is_finished": bool,         # Termination flag
}
```

## 📁 Project Structure

```
PSC_Agents/
├── src/
│   ├── agent/                    # Agent implementations
│   │   ├── meta_agent.py         # Strategic planning (no tools)
│   │   ├── data_agent.py         # Literature search (MCP)
│   │   ├── design_agent.py       # Material design (Server tools)
│   │   ├── fab_agent.py          # Performance prediction (Local RF)
│   │   ├── analysis_agent.py     # Gap analysis (Chemistry tools)
│   │   └── memory_agent.py       # Knowledge archival (no tools)
│   │
│   ├── core/                     # Core infrastructure
│   │   ├── base_agent.py         # BaseAgent with LLM & MCP support
│   │   ├── config.py             # Settings & configuration
│   │   ├── llm.py                # LLM client wrapper
│   │   └── tool.py               # Tool registry
│   │
│   ├── workflow/                 # Workflow orchestration
│   │   ├── graph.py              # LangGraph workflow definition
│   │   └── state.py              # AgentState TypedDict
│   │
│   └── test/                     # Test scripts
│       ├── workflow_test.py      # Full workflow test
│       ├── design_agent_experiment.py
│       ├── fab_agent_experiment.py
│       └── analysis_agent_experiment.py
│
├── mcp/                          # MCP tool implementations
│   ├── analysis_agent/           # Chemistry & visualization tools
│   ├── data_agent/               # arXiv MCP server
│   ├── design_agent/             # Server tools (MatterGen, CSLLM)
│   └── fab_agent/                # RF prediction models
│
├── dataset/                      # Training data & models
│   ├── CSLLM/                    # CSLLM inference code
│   └── mattergen/                # MatterGen dataset
│
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/PSC_Agents.git
cd PSC_Agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

# Optional: Project settings
PROJECT_NAME=PSC_Agents
```

### 3. Run Workflow Test

```bash
cd src/test

# Full workflow with MCP tools
python workflow_test.py

# Mock mode (no external dependencies)
python workflow_test.py --mode mock

# Custom research goal
python workflow_test.py --query "Design PCE > 25% perovskite"

# Show available test queries
python workflow_test.py --list-queries
```

## 🔧 Agent Details

### MetaAgent (Chief Scientist)

**Role**: Pure reasoning agent that orchestrates the research workflow.

**Capabilities**:
- Parse user goals and identify constraints
- Formulate testable scientific hypotheses
- Assign specific tasks to downstream agents
- Decide: CONTINUE, SKIP, REDO, PIVOT, TERMINATE

**Input State**:
- `goal`: User's research objective
- `memory_log`: Knowledge from previous iterations
- `analysis_report`: Latest diagnosis

**Output**: `plan` (JSON with hypothesis, strategy, agent_tasks)

---

### DataAgent (Literature Intelligence)

**Role**: Search, download, and extract data from scientific papers.

**Tools** (MCP - arXiv Server):
- `search_papers(query, max_results)`: Search arXiv
- `download_paper(paper_id)`: Download by ID
- `read_paper(paper_id)`: Convert PDF to text

**Local Tools**:
- `save_markdown_locally(path)`: Save paper to disk
- `extract_data_from_papers(goal, plan, dir)`: LLM extraction

**Output**: `data_context` (JSON with findings, extracted data)

---

### DesignAgent (Material Design Expert)

**Role**: Design perovskite materials and synthesis routes.

**Tools** (Server - MatterGen & CSLLM):
- `generate_material_structure`: Generate candidate structures
- `check_synthesizability`: Verify if formula can be synthesized
- `predict_synthesis_method`: Predict optimal synthesis route
- `predict_precursors`: Identify precursor chemicals

**Tool Modes**:
- `mock`: Simulated results for testing
- `interactive`: Wait for server input via terminal

**Output**: `experimental_params` (JSON with composition, process, precursors)

---

### FabAgent (Virtual Fabrication)

**Role**: Predict solar cell performance using trained ML models.

**Tools** (Local RF Models):
- `predict_perovskite(composition)`: Predict PCE, Voc, Jsc, FF, etc.
- `visualize_predictions`: Bar chart visualization
- `visualize_series_trend`: Line chart for trends
- `visualize_comparison`: Grouped bar chart comparison

**Output**: `fab_results` (JSON with predicted metrics)

---

### AnalysisAgent (Lead Analyst)

**Role**: Diagnose performance gaps and identify root causes.

**Tools**:
- `analyze_stoichiometry`: Chemical formula validation (pymatgen)
- `analyze_organic_cation`: Cation properties (RDKit)
- `analyze_mechanism`: Degradation/performance diagnosis
- `calculate_correlation`: Statistical correlation analysis
- `shap_feature_importance`: ML interpretability
- `visualize_structure`: 3D crystal visualization

**Output**: `analysis_report` (JSON with diagnosis, gap analysis)

---

### MemoryAgent (Knowledge Keeper)

**Role**: Archive insights and maintain long-term memory.

**Mission**:
- Extract knowledge triplet (formula, PCE, reason)
- Identify why results succeeded/failed
- Detect trends across iterations

**Output**: `memory_log` (append structured entry)

## 🧪 Running Experiments

### Single Agent Experiments

```bash
cd src/test

# DesignAgent with 40 queries
python design_agent_experiment.py --mode mock

# FabAgent with prediction tasks
python fab_agent_experiment.py

# AnalysisAgent with 20 queries
python analysis_agent_experiment.py
```

### Full Workflow Experiments

```bash
# Multiple iterations
python workflow_test.py --iterations 5

# Save results
python workflow_test.py --output-dir results/exp1

# Different queries
python workflow_test.py --query-index 0  # PCE > 20%
python workflow_test.py --query-index 1  # Lead-free
python workflow_test.py --query-index 2  # Mixed cation
```

## 🔌 Server Tools Integration

DesignAgent supports interactive mode for server tools (MatterGen, CSLLM):

```python
# In experiment script
agent = DesignAgent(tool_mode="interactive")

# When tool is called, you'll see:
# ════════════════════════════════════════════════════════════
# 📡 SERVER TOOL CALL: check_synthesizability
# ════════════════════════════════════════════════════════════
# 📥 Input:
# {
#   "formula": "Cs0.05FA0.95PbI3"
# }
# 
# 🖥️  Run on server: csllm_inference.py --mode synth --formula "Cs0.05FA0.95PbI3"
# 
# 📤 Paste result JSON below (type 'END' on new line when done):

# Then paste your server output and type END
```

## 📊 Supported Research Tasks

The system can handle various perovskite research queries:

| Task Type | Example Query |
|-----------|---------------|
| **Composition Design** | "Design PCE > 25% perovskite composition" |
| **Lead-Free** | "Design Sn-based lead-free perovskite" |
| **Stability** | "Optimize for thermal stability > 85°C" |
| **Bandgap Engineering** | "Design 1.7 eV bandgap for tandem cells" |
| **Mixed Cation** | "Optimize FA/MA/Cs ratio for stability" |
| **2D Perovskites** | "Design 2D Ruddlesden-Popper structure" |
| **Analysis** | "Analyze Voc loss mechanism in CsPbI3" |

## 🛠️ Development

### Adding New Tools

1. Define tool in agent's tools list:
```python
NEW_TOOL = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "...",
        "parameters": {...}
    }
}
```

2. Implement execution function:
```python
async def _execute_my_tool(self, args: dict) -> dict:
    # Implementation
    return {"result": "..."}
```

3. Register in executor map:
```python
TOOL_EXECUTORS = {
    "my_tool": _execute_my_tool,
}
```

### Customizing Agent Prompts

Each agent's behavior is controlled by its `SYSTEM_PROMPT`:

```python
# In agent file
SYSTEM_PROMPT = """You are [AgentName] - [Role] of PSC_Agents.

## Role
...

## Your Input (State Access)
- `goal`: ...
- `plan`: ...

## Your Specialized Toolbox
- `tool1`: Description
- `tool2`: Description

## Output
Your output -> `state_field`
"""
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- LangGraph for workflow orchestration
- arXiv for paper access
- MatterGen & CSLLM for material generation
- pymatgen & RDKit for chemistry tools

## 📧 Contact

For questions or collaboration, please open an issue or contact the team.
