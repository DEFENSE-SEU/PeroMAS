"""
FabAgent - Virtual Fabrication & Simulation Engineer for PSC_Agents.

Responsible for executing the "Make & Measure" step via predictive models.
It takes the design parameters and uses AI/Physics surrogates to predict 
performance (PCE) and stability (T80).

Author: PSC_Agents Team
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.base_agent import BaseAgent
from core.config import Settings


# === Type-safe Helper Functions ===
def safe_str(value: Any, default: str = "") -> str:
    """Safely convert any value to string, handling None, list, dict."""
    if value is None:
        return default
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else default
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False) if value else default
    return str(value)


def safe_truncate(value: Any, max_len: int, suffix: str = "...", default: str = "N/A") -> str:
    """Safely truncate any value to max_len characters."""
    str_value = safe_str(value, default)
    if len(str_value) > max_len:
        return str_value[:max_len] + suffix
    return str_value


# Import visualization tool - handle multiple possible import paths
visualize_prediction_results = None
PredictionVisualizer = None

# Try different import paths
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

try:
    from mcp.fab_agent.visualization import visualize_prediction_results, PredictionVisualizer
except ImportError:
    # Try direct path import as fallback
    _viz_path = _project_root / "mcp" / "fab_agent"
    if _viz_path.exists():
        sys.path.insert(0, str(_viz_path))
        try:
            from visualization import visualize_prediction_results, PredictionVisualizer
        except ImportError:
            pass

# Import Perovskite predictor
predict_perovskite_properties = None
PerovskitePredictor = None

try:
    from mcp.fab_agent.perovskite_predictor import predict_perovskite_properties, PerovskitePredictor, ALL_TARGETS, TARGET_INFO
except ImportError:
    try:
        from perovskite_predictor import predict_perovskite_properties, PerovskitePredictor, ALL_TARGETS, TARGET_INFO
    except ImportError:
        ALL_TARGETS = ["pce", "voc", "jsc", "ff", "dft_band_gap", "energy_above_hull"]
        TARGET_INFO = {}


# === Local Tools Definition ===
FAB_AGENT_TOOLS = [
    {
        "name": "predict_perovskite",
        "description": "Predict perovskite solar cell properties using trained RF models. Supports both composition formula and CIF structure input. Predicts: PCE (%), Voc (V), Jsc (mA/cm2), FF (%), Band Gap (eV), Energy Above Hull (eV/atom).",
        "parameters": {
            "type": "object",
            "properties": {
                "composition": {
                    "type": "string",
                    "description": "Perovskite composition formula, e.g., 'CsPbI3', 'MAPbI3', 'FA0.25MA0.75PbI3', 'Cs0.05FA0.79MA0.16Pb(I0.83Br0.17)3'"
                },
                "cif_content": {
                    "type": "string",
                    "description": "CIF file content string for crystal structure input"
                },
                "cif_file": {
                    "type": "string",
                    "description": "Path to CIF file"
                },
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific targets to predict. Options: pce, voc, jsc, ff, dft_band_gap, energy_above_hull. Default: all"
                }
            },
            "required": []
        }
    },
    {
        "name": "visualize_predictions",
        "description": "Create bar chart visualization of predicted performance metrics (PCE, Voc, Jsc, FF) for a single material. Outputs PNG and HTML files.",
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_metrics": {
                    "type": "object",
                    "description": "Dict of metric values: {PCE_percent, Voc_V, Jsc_mA_cm2, FF_percent, BandGap_eV}"
                },
                "target_metrics": {
                    "type": "object",
                    "description": "Optional target values for comparison bars"
                },
                "recipe_id": {
                    "type": "string",
                    "description": "Identifier for this prediction batch"
                }
            },
            "required": ["predicted_metrics"]
        }
    },
    {
        "name": "visualize_series_trend",
        "description": """Create a trend line chart showing how a property changes across a series of compositions.

IMPORTANT: Each item in series_data MUST include the full 'predictions' dict from predict_perovskite.

Example usage:
1. Call predict_perovskite for each composition
2. Build series_data with predictions included:
   series_data = [
     {"x_value": 0, "x_label": "FAPbI3", "predictions": {"pce": {"value": 20.1}, "voc": {"value": 1.05}, ...}},
     {"x_value": 0.1, "x_label": "FA0.9Cs0.1PbI3", "predictions": {"pce": {"value": 21.2}, "voc": {"value": 1.08}, ...}},
   ]
3. Call visualize_series_trend with series_data and y_metric""",
        "parameters": {
            "type": "object",
            "properties": {
                "series_data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x_value": {"type": "number", "description": "X-axis numerical value (e.g., 0, 0.1, 0.2)"},
                            "x_label": {"type": "string", "description": "X-axis label (material name)"},
                            "predictions": {"type": "object", "description": "REQUIRED: Full predictions dict from predict_perovskite, e.g., {'pce': {'value': 20.1}, 'voc': {'value': 1.05}, ...}"}
                        },
                        "required": ["x_value", "x_label", "predictions"]
                    },
                    "description": "Array of data points. EACH item MUST have predictions dict with the metric values."
                },
                "x_label": {
                    "type": "string",
                    "description": "X-axis title, e.g., 'Cs Content (x)', 'Br Ratio'"
                },
                "y_metric": {
                    "type": "string",
                    "enum": ["pce", "voc", "jsc", "ff", "dft_band_gap", "energy_above_hull"],
                    "description": "Which metric to plot on Y-axis"
                },
                "title": {
                    "type": "string",
                    "description": "Chart title, e.g., 'PCE vs Cs Content in FA(1-x)Cs(x)PbI3'"
                }
            },
            "required": ["series_data", "y_metric", "title"]
        }
    },
    {
        "name": "visualize_comparison",
        "description": """Create a grouped bar chart comparing multiple materials across selected metrics.

IMPORTANT: Each item in materials_data MUST include the full 'predictions' dict from predict_perovskite.

Example usage:
1. Call predict_perovskite for each material
2. Build materials_data with predictions included:
   materials_data = [
     {"name": "MAPbI3", "predictions": {"pce": {"value": 19.1}, "voc": {"value": 1.05}, ...}},
     {"name": "FAPbI3", "predictions": {"pce": {"value": 20.5}, "voc": {"value": 1.08}, ...}},
   ]
3. Call visualize_comparison with materials_data""",
        "parameters": {
            "type": "object",
            "properties": {
                "materials_data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Material name (e.g., 'MAPbI3', 'FAPbI3')"},
                            "predictions": {"type": "object", "description": "REQUIRED: Full predictions dict from predict_perovskite"}
                        },
                        "required": ["name", "predictions"]
                    },
                    "description": "Array of materials. EACH item MUST have predictions dict with metric values."
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metrics to compare, e.g., ['pce', 'voc', 'jsc', 'ff']. Default: all four"
                },
                "title": {
                    "type": "string",
                    "description": "Chart title, e.g., 'Performance Comparison: MAPbI3 vs FAPbI3 vs CsPbI3'"
                }
            },
            "required": ["materials_data", "title"]
        }
    }
]


# === System Prompt ===
SYSTEM_PROMPT = """You are FabAgent - Virtual Fabrication Engineer of PSC_Agents.

## Core Mission
Predict perovskite solar cell performance using trained ML models. Autonomously decide which tools to use based on the specific request.

## Your Specialized Toolbox
- `predict_perovskite`: Predict material properties from composition formula or CIF. Returns: PCE, Voc, Jsc, FF, Band Gap, E_hull.
- `visualize_predictions`: Bar chart for single material. **Constraint:** Only when user requests visualization.
- `visualize_series_trend`: Line chart for property trends. **Constraint:** Only when user requests trend visualization.
- `visualize_comparison`: Grouped bar chart comparing materials. **Constraint:** Only when user requests comparison chart.

## Tool Constraints
1. **Prediction Before Visualization**: Call `predict_perovskite` before any visualization tool.
2. **Visualization on Request Only**: Don't visualize unless explicitly asked for charts/plots/graphs.
3. **Data Format**: Visualization tools need the full prediction results from `predict_perovskite`.

## Operational Guidelines
- **Single Prediction**: Predict one material → Return metrics and interpretation
- **Multi-Material**: Predict each material → Compare and summarize
- **Trend Analysis**: Predict series → Describe trends in text
- **Visualization**: Only when requested → Call appropriate viz tool after predictions

## Output Principles
- Base output on actual tool results, not assumptions
- Include prediction metrics as returned by the model
- Provide scientific interpretation relevant to the goal
- Status reflects whether prediction succeeded
"""


class FabAgent(BaseAgent):
    """
    Virtual Fabrication Engineer agent.

    Interfaces:
        Inherits 'autonomous_thinking' to access MCP tools.

    Expected MCP Tools:
        - performance_predictor: ML model (e.g., Random Forest/GNN) for efficiency.
        - stability_predictor: ML model for lifetime prediction.
    
    Local Tools:
        - predict_perovskite: Predict solar cell properties (PCE, Voc, Jsc, FF, Band Gap, E_hull)
          from composition formula or CIF structure using trained RF models
        - visualize_predictions: Bar chart visualization of predicted metrics
    """

    def __init__(self, settings: Settings | None = None, output_dir: str | None = None) -> None:
        """
        Initialize the FabAgent.
        
        Args:
            settings: Configuration settings
            output_dir: Output directory for visualizations
        """
        super().__init__(name="FabAgent", settings=settings)
        
        # Register local tools
        self._local_tools = FAB_AGENT_TOOLS
        self._output_dir = output_dir or "output"
        self._current_query_id: str | None = None

    def _get_local_tools(self) -> list[dict[str, Any]]:
        """Return local tool definitions for this agent."""
        return self._local_tools
    
    def set_query_id(self, query_id: str) -> None:
        """Set current query ID for file naming."""
        self._current_query_id = query_id
    
    def set_output_dir(self, output_dir: str) -> None:
        """Set output directory for visualizations."""
        self._output_dir = output_dir

    async def _get_tools_with_local_tools(self) -> list[dict[str, Any]]:
        """Get tool list including local tools and MCP tools."""
        # Get MCP tools if available
        mcp_tools = []
        if self.registry.is_initialized():
            mcp_tools = await self.registry.get_tools_schema()
        
        # Convert local tools to proper format
        local_tools = []
        for tool in self._local_tools:
            if "name" in tool:
                # Already in simple format
                local_tools.append({"type": "function", "function": tool})
            else:
                local_tools.append(tool)
        
        return local_tools + mcp_tools

    async def _execute_local_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a local tool by name."""
        if tool_name == "predict_perovskite":
            return await self._execute_predict_perovskite(arguments)
        elif tool_name == "visualize_predictions":
            return await self._execute_visualize_predictions(arguments)
        elif tool_name == "visualize_series_trend":
            return await self._execute_visualize_series_trend(arguments)
        elif tool_name == "visualize_comparison":
            return await self._execute_visualize_comparison(arguments)
        else:
            return {"error": f"Unknown local tool: {tool_name}"}

    async def _execute_predict_perovskite(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute the predict_perovskite tool.
        
        Predicts perovskite solar cell properties using trained RF models.
        """
        if predict_perovskite_properties is None:
            return {
                "status": "error",
                "message": "Perovskite predictor not available. Check mcp/fab_agent/perovskite_predictor.py"
            }
        
        composition = arguments.get("composition")
        cif_content = arguments.get("cif_content")
        cif_file = arguments.get("cif_file")
        targets = arguments.get("targets")
        
        # At least one input is required.
        if not composition and not cif_content and not cif_file:
            return {
                "status": "error",
                "message": "No input provided. Please specify composition, cif_content, or cif_file."
            }
        
        try:
            result = predict_perovskite_properties(
                composition=composition,
                cif_content=cif_content,
                cif_file=cif_file,
                targets=targets
            )
            
            # Add status info.
            if "predictions" in result:
                result["status"] = "success"
                # Print a brief prediction summary.
                print(f"\n   📊 Prediction Results for: {result.get('input', 'unknown')}")
                for target, pred in result["predictions"].items():
                    if isinstance(pred, dict) and "value" in pred:
                        print(f"      {pred.get('name', target)}: {pred['value']:.4f} {pred.get('unit', '')}")
            else:
                result["status"] = "error"
            
            return result
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _execute_visualize_predictions(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the visualize_predictions tool."""
        if visualize_prediction_results is None:
            return {
                "status": "error",
                "message": "Visualization module not available. Install matplotlib or plotly."
            }
        
        predicted_metrics = arguments.get("predicted_metrics", {})
        target_metrics = arguments.get("target_metrics")
        recipe_id = arguments.get("recipe_id", "Prediction_Batch")
        
        try:
            result = visualize_prediction_results(
                predicted_metrics=predicted_metrics,
                target_metrics=target_metrics,
                recipe_id=recipe_id,
                output_dir=self._output_dir,
            )
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _execute_visualize_series_trend(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute the visualize_series_trend tool.
        
        Creates a trend line chart for composition series.
        """
        try:
            from mcp.fab_agent.visualization import visualize_series_trend
        except ImportError:
            try:
                from visualization import visualize_series_trend
            except ImportError:
                return {
                    "status": "error",
                    "message": "visualize_series_trend not available. Check mcp/fab_agent/visualization.py"
                }
        
        series_data = arguments.get("series_data", [])
        x_label = arguments.get("x_label", "Composition Parameter")
        y_metric = arguments.get("y_metric", "pce")
        title = arguments.get("title", "Property Trend")
        
        # Prefix title with query_id.
        if self._current_query_id:
            title = f"{self._current_query_id}_{title}"
        
        if not series_data:
            return {
                "status": "error",
                "message": "No series_data provided. Please provide array of {x_value, x_label, predictions}."
            }
        
        try:
            result = visualize_series_trend(
                series_data=series_data,
                x_label=x_label,
                y_metric=y_metric,
                title=title,
                output_dir=self._output_dir,
            )
            print(f"\n   [Chart] Trend chart saved: {result.get('file_path') or result.get('png_path')}")
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _execute_visualize_comparison(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute the visualize_comparison tool.
        
        Creates a grouped bar chart comparing multiple materials.
        """
        try:
            from mcp.fab_agent.visualization import visualize_comparison
        except ImportError:
            try:
                from visualization import visualize_comparison
            except ImportError:
                return {
                    "status": "error",
                    "message": "visualize_comparison not available. Check mcp/fab_agent/visualization.py"
                }
        
        materials_data = arguments.get("materials_data", [])
        metrics = arguments.get("metrics")
        title = arguments.get("title", "Materials Comparison")
        
        # Prefix title with query_id.
        if self._current_query_id:
            title = f"{self._current_query_id}_{title}"
        
        if not materials_data:
            return {
                "status": "error",
                "message": "No materials_data provided. Please provide array of {name, predictions}."
            }
        
        try:
            result = visualize_comparison(
                materials_data=materials_data,
                metrics=metrics,
                title=title,
                output_dir=self._output_dir,
            )
            print(f"\n   [Chart] Comparison chart saved: {result.get('file_path') or result.get('png_path')}")
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_system_prompt(
        self,
        state: dict[str, Any],
        default_prompt: str | None = None,
    ) -> str:
        """Return the domain-specific system prompt."""
        return SYSTEM_PROMPT

    async def autonomous_thinking(
        self,
        prompt: str,
        state: dict[str, Any],
        system_message: str | None = None,
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Override autonomous_thinking to handle local tools.
        """
        if not self.llm:
            self.logger.error("LLM client not available")
            return {
                "response": "[ERROR] LLM not configured",
                "tool_calls": [],
                "tool_results": [],
                "iterations": 0,
            }

        # Get tools (including local tools)
        tools = await self._get_tools_with_local_tools()
        local_tool_names = {t.get('name', t.get('function', {}).get('name', '')) for t in self._local_tools}
        
        self.logger.info(f"Available tools: {len(tools)}")
        tool_names = [t.get('function', {}).get('name', t.get('name', 'unknown')) for t in tools]
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
                
                # Tool call visualization with de-duplication.
                tool_type = "📍 Local" if tool_name in local_tool_names else "🌐 MCP"
                if tool_name == _last_tool_name:
                    _consecutive_count += 1
                    print(f"\r   🔄 [FabAgent] {tool_name} called {_consecutive_count}x (consecutive)", end="", flush=True)
                else:
                    if _last_tool_name is not None and _consecutive_count > 1:
                        print()  # End the previous tool's counter line.
                    _consecutive_count = 1
                    _last_tool_name = tool_name
                    print(f"\n🔧 [FabAgent] Calling {tool_type} Tool: {tool_name}")
                    print(f"   📥 Arguments: {str(tool_args)[:200]}{'...' if len(str(tool_args)) > 200 else ''}")
                
                all_tool_calls.append(tc)

                try:
                    # Check if it's a local tool
                    if tool_name in local_tool_names:
                        result = await self._execute_local_tool(tool_name, tool_args)
                        result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    else:
                        # MCP tool
                        result = await self.registry.call_tool(tool_name, tool_args)
                        result_str = str(result) if result else "No result"
                    
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

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute virtual fabrication based on MetaAgent's task.
        """
        print(f"\n{'='*60}")
        print(f"🏭 [FabAgent] Virtual Fabrication")
        print(f"{'='*60}")
        
        # Show available tools with categorization
        tools = await self._get_tools_with_local_tools()
        local_tool_names = [t.get('name', t.get('function', {}).get('name', 'unknown')) for t in self._local_tools]
        mcp_tool_names = []
        for t in tools:
            name = t.get('function', {}).get('name', t.get('name', 'unknown'))
            if name not in local_tool_names:
                mcp_tool_names.append(name)
        
        print(f"\n🛠️  Available Tools Summary:")
        print(f"   📍 Local Tools ({len(local_tool_names)}): {local_tool_names}")
        print(f"   🌐 MCP Tools ({len(mcp_tool_names)}): {mcp_tool_names if mcp_tool_names else 'None'}")
        print(f"   📊 Total: {len(tools)} tools")
        
        # Get context from state - include ALL upstream outputs
        goal = state.get("goal", "")
        plan = state.get("plan", "")
        data_context = state.get("data_context", "")  # From DataAgent
        experimental_params = state.get("experimental_params", {})  # From DesignAgent
        
        # Extract FabAgent-specific task from MetaAgent's plan
        my_task = self._extract_my_task(plan, "FabAgent")
        
        # === Display upstream context clearly ===
        print(f"\n📊 Upstream Context:")
        print(f"   ├─ 🎯 Goal: {safe_truncate(goal, 80)}")
        print(f"   ├─ 📝 Task: {my_task}")
        print(f"   ├─ 📚 Data (DataAgent): {len(safe_str(data_context))} chars")
        
        # Extract key info from DesignAgent output
        if experimental_params:
            composition = experimental_params.get("composition", {})
            formula = safe_str(composition.get("formula"), "N/A") if isinstance(composition, dict) else "N/A"
            process = experimental_params.get("process", {})
            method = safe_str(process.get("method"), "N/A") if isinstance(process, dict) else "N/A"
            protocol = safe_str(process.get("synthesis_protocol")) if isinstance(process, dict) else ""
            print(f"   └─ 🧪 Design (DesignAgent):")
            print(f"       ├─ Formula: {formula}")
            print(f"       ├─ Method: {method}")
            if protocol:
                print(f"       └─ Protocol: {safe_truncate(protocol, 150)}")
            else:
                print(f"       └─ Protocol: N/A")
        else:
            print(f"   └─ 🧪 Design: None")

        if not experimental_params:
            return {"fab_results": {
                "status": "failed",
                "error": "No experimental parameters from DesignAgent",
                "composition": "N/A",
                "predicted_metrics": {}  # Empty dict, not None - safer for downstream processing
            }}

        recipe_str = json.dumps(experimental_params, indent=2)

        prompt = f"""# 🏭 VIRTUAL FABRICATION MISSION
**Assigned Task**: {my_task}
**Research Context**: {goal}

# 📚 LITERATURE CONTEXT (from DataAgent)
{safe_truncate(data_context, 5000, default='No literature data available.')}

# 🧪 INPUT EXPERIMENTAL RECIPE (from DesignAgent)
```json
{recipe_str}
```

# 🎯 YOUR MISSION
Extract the formula from the recipe and predict its properties using `predict_perovskite`.
Interpret the results in context of the research goal.

# 📋 OUTPUT REQUIREMENTS
After calling prediction tools, provide results as JSON including:
- The composition you predicted
- The predicted metrics from the tool
- Your analysis of how results compare to the goal
- Status of the prediction

Only report actual tool outputs - don't invent values.
"""

        result = await self.autonomous_thinking(
            prompt=prompt,
            state=state,
            system_message=SYSTEM_PROMPT,
            max_iterations=6,
        )

        response_text = result.get("response", "")
        tool_results = result.get("tool_results", [])
        fab_results = self._build_fab_results(response_text, tool_results, experimental_params)

        # === Display output summary ===
        print(f"\n{'─'*60}")
        print(f"✅ [FabAgent] Fabrication Complete")
        print(f"   ├─ Tool calls: {len(tool_results)}")
        if fab_results.get("predicted_metrics"):
            metrics = fab_results["predicted_metrics"]
            print(f"   ├─ Composition: {fab_results.get('composition', 'N/A')}")
            print(f"   ├─ PCE: {metrics.get('PCE_percent', 'N/A')}%")
            print(f"   └─ Status: {fab_results.get('status', 'N/A')}")

        return {"fab_results": fab_results}
    
    def _extract_my_task(self, plan: str | dict, agent_name: str) -> str:
        """Extract specific task for this agent from MetaAgent's plan."""
        if not plan:
            return "Run performance and stability predictions"
        
        # If plan is already a dict, use it directly
        if isinstance(plan, dict):
            plan_data = plan
        else:
            # Try to parse JSON from string plan
            try:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', str(plan))
                if match:
                    plan_data = json.loads(match.group(1))
                elif '{' in str(plan):
                    plan_str = str(plan)
                    start = plan_str.find('{')
                    end = plan_str.rfind('}') + 1
                    plan_data = json.loads(plan_str[start:end])
                else:
                    return safe_truncate(plan, 500)
            except (json.JSONDecodeError, KeyError):
                return safe_truncate(plan, 500)
        
        # Extract agent-specific task
        agent_tasks = plan_data.get("agent_tasks", {})
        task = agent_tasks.get(agent_name, "")
        
        if task and str(task).upper() != "SKIP":
            return task
        else:
            return plan_data.get("iteration_focus", "Run predictions")

    def _build_fab_results(
        self,
        response: str,
        tool_results: list[dict[str, Any]],
        experimental_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Construct a standardized result dictionary.
        
        Output Schema:
        {
            "composition": "formula",
            "predicted_metrics": {
                "PCE_percent": value,
                "Voc_V": value,
                "Jsc_mA_cm2": value,
                "FF_percent": value,
                "BandGap_eV": value,
                "E_hull_eV": value
            },
            "analysis": "interpretation",
            "recommendation": "advice for next agent",
            "status": "success/failed"
        }
        """
        # Try to extract structured data from LLM response
        parsed = self._extract_json_block(response)
        
        # Ensure parsed is a dict (may be a list or other type)
        if parsed is not None and not isinstance(parsed, dict):
            parsed = None
        
        # Get composition from experimental_params if not in parsed
        composition = None
        if parsed and isinstance(parsed, dict) and parsed.get("composition"):
            composition = parsed.get("composition")
        elif experimental_params:
            composition = experimental_params.get("composition", {}).get("formula")
        
        # Extract metrics from tool results if not in parsed
        predicted_metrics = None
        if parsed and parsed.get("predicted_metrics"):
            predicted_metrics = parsed.get("predicted_metrics")
        else:
            # Try to find metrics in tool results
            for tr in tool_results:
                if tr.get("tool") == "predict_perovskite":
                    try:
                        result_data = json.loads(tr.get("result", "{}"))
                        if "predictions" in result_data:
                            preds = result_data["predictions"]
                            predicted_metrics = {
                                "PCE_percent": preds.get("pce", {}).get("value"),
                                "Voc_V": preds.get("voc", {}).get("value"),
                                "Jsc_mA_cm2": preds.get("jsc", {}).get("value"),
                                "FF_percent": preds.get("ff", {}).get("value"),
                                "BandGap_eV": preds.get("bandgap", {}).get("value"),
                                "E_hull_eV": preds.get("e_hull", {}).get("value"),
                            }
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        return {
            "composition": composition,
            "predicted_metrics": predicted_metrics,
            "analysis": parsed.get("analysis") if parsed else safe_truncate(response, 500),
            "recommendation": parsed.get("recommendation") if parsed else None,
            "status": "success" if predicted_metrics else "failed",
            "raw_response": response,
            "tool_calls": len(tool_results),
        }

    def _extract_json_block(self, text: str) -> Any | None:
        """Robustly extract JSON from Markdown."""
        try:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return json.loads(match.group(1))
            
            # Fallback: find first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
                
        except Exception:
            pass
        return None
