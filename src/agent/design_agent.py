"""
DesignAgent - Experimental Design Expert for PSC_Agents.

Responsible for translating research goals and literature data into
precise fabrication recipes:
1. Material Composition & Structure Design (via MatterGen)
2. Synthesizability Check & Process Parameter Optimization (via CSLLM)

Supports two execution modes:
- MOCK: Use simulated results for local testing
- INTERACTIVE: Wait for server results via terminal input

Author: PSC_Agents Team
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, ClassVar, Optional

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mcp" / "design_agent"))

from core.base_agent import BaseAgent
from core.config import Settings
from server_tools import ServerToolManager, ToolMode


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

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are DesignAgent - Experimental Design Expert of PSC_Agents.

## Core Mission
Design perovskite materials and plan synthesis routes. Autonomously decide which tools to use based on the specific request.

## Your Specialized Toolbox

### Material Generation (Server - MatterGen)
- `generate_material_structure`: Generate candidate perovskite structures with target properties.
  - Input: target_pce, target_bandgap, stability_threshold, num_candidates
  - Output: Multiple candidate structures with predicted properties

### Synthesis Prediction (Server - CSLLM)
- `check_synthesizability`: Verify if a formula can be synthesized.
  - Input: formula, structure_type (2D/3D)
  - Output: synthesizable, confidence, reasoning

- `predict_synthesis_method`: Predict optimal synthesis route.
  - Input: formula
  - Output: method, confidence, reasoning

- `predict_precursors`: Identify precursor chemicals and solvents.
  - Input: formula, synthesis_method
  - Output: precursors list, solvents, concentrations

### Local Tools
- `screen_candidates`: Filter and rank multiple candidates by criteria.
  - Input: candidates list, criteria dict
  - Output: Ranked candidates with recommendations

## ⚠️ CRITICAL CONSTRAINTS

### 1. **MUST USE ACTUAL TOOL RESULTS**
- When `generate_material_structure` returns formulas, YOU MUST USE THOSE EXACT FORMULAS for subsequent tool calls.
- DO NOT invent or substitute formulas. The server-generated formulas are the only valid candidates.
- Example: If server returns "MAPbBrI2", you MUST call `check_synthesizability(formula="MAPbBrI2")`, NOT some other formula.

### 2. **Tool Dependencies**
- `predict_precursors` requires knowing the synthesis_method first
- For checking synthesizability of generated materials, use the EXACT formulas from generation output

### 3. **Server Tool Behavior**
- Server tools return real experimental/computational data
- Trust the server results - they represent actual scientific knowledge
- If a formula cannot be parsed or is invalid, report that finding

## Operational Guidelines
- **Design Material**: Generate candidates, then check/screen using the EXACT generated formulas
- **Check Synthesizability**: Just call `check_synthesizability`
- **Get Synthesis Method**: Just call `predict_synthesis_method`
- **Get Precursors**: May need method first, then call `predict_precursors`
- **Full Recipe**: Combine tools as needed, always using actual results from previous steps

## Output Principles
- Base output STRICTLY on actual tool results - never invent data
- When reporting formulas, use the EXACT formulas from tool outputs
- Provide scientific reasoning for design choices
- If tool returns unexpected/problematic results, analyze and report honestly
"""


class DesignAgent(BaseAgent):
    """
    Experimental Design Expert agent.
    
    Uses MatterGen for material structure generation and CSLLM for
    synthesis prediction. Supports two modes:
    - MOCK: Use simulated results for testing
    - INTERACTIVE: Wait for server results via terminal input
    """

    # Documentation: Tool types used by this agent
    REQUIRED_TOOL_TYPES: ClassVar[list[str]] = [
        "material_structure_generation",  # MatterGen
        "synthesizability_prediction",    # CSLLM synthesis
        "synthesis_method_prediction",    # CSLLM method
        "precursor_prediction",           # CSLLM precursor
    ]
    
    # Server tool names (handled by ServerToolManager)
    SERVER_TOOLS = {
        "generate_material_structure",
        "check_synthesizability",
        "predict_synthesis_method",
        "predict_precursors",
    }
    
    # Local tool names (handled internally)
    LOCAL_TOOLS = {
        "screen_candidates",
    }

    def __init__(
        self,
        settings: Settings | None = None,
        tool_mode: str = "mock"
    ) -> None:
        """
        Initialize the DesignAgent.
        
        Args:
            settings: Configuration settings
            tool_mode: "mock" for simulated results, "interactive" for server input
        """
        super().__init__(name="DesignAgent", settings=settings)
        
        # Initialize server tool manager
        self._server_tools = ServerToolManager(mode=tool_mode)
        self._tool_mode = tool_mode
        
        # Store candidates for screening
        self._current_candidates: list[dict] = []
        
        self.logger.info(f"DesignAgent initialized with tool_mode={tool_mode}")

    def set_tool_mode(self, mode: str) -> None:
        """
        Set the tool execution mode.
        
        Args:
            mode: "mock" or "interactive"
        """
        self._tool_mode = mode
        self._server_tools.set_mode(mode)
        self.logger.info(f"Tool mode set to: {mode}")
    
    def _screen_candidates(self, candidates: list[dict], criteria: dict) -> dict:
        """
        Screen and rank candidates based on multi-criteria scoring.
        
        Args:
            candidates: List of candidate materials with properties
            criteria: Screening criteria and weights
            
        Returns:
            Screening results with ranked candidates
        """
        if not candidates:
            return {
                "status": "error",
                "message": "No candidates provided for screening"
            }
        
        # Extract weights from criteria
        pce_weight = criteria.get("pce_weight", 0.4)
        stability_weight = criteria.get("stability_weight", 0.3)
        synthesizability_weight = criteria.get("synthesizability_weight", 0.3)
        min_pce = criteria.get("min_pce", 0.0)
        max_energy_above_hull = criteria.get("max_energy_above_hull", 0.1)
        
        ranked_candidates = []
        
        for i, candidate in enumerate(candidates):
            # Extract properties with defaults
            pce = candidate.get("predicted_pce", 0)
            energy_above_hull = candidate.get("energy_above_hull", 0.1)
            confidence = candidate.get("confidence", 0.5)
            formula = candidate.get("formula", f"Candidate_{i}")
            
            # Apply filters
            if pce < min_pce:
                continue
            if energy_above_hull > max_energy_above_hull:
                continue
            
            # Normalize scores (0-1)
            pce_score = min(pce / 30.0, 1.0)  # Assume max PCE ~30%
            stability_score = max(0, 1 - energy_above_hull / 0.1)  # Lower is better
            synth_score = confidence
            
            # Compute weighted score
            total_score = (
                pce_weight * pce_score +
                stability_weight * stability_score +
                synthesizability_weight * synth_score
            )
            
            ranked_candidates.append({
                "rank": 0,  # Will be set after sorting
                "formula": formula,
                "score": round(total_score, 3),
                "scores_breakdown": {
                    "pce_score": round(pce_score, 3),
                    "stability_score": round(stability_score, 3),
                    "synthesizability_score": round(synth_score, 3)
                },
                "properties": {
                    "predicted_pce": pce,
                    "energy_above_hull": energy_above_hull,
                    "confidence": confidence
                },
                "original_data": candidate
            })
        
        # Sort by score descending
        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Assign ranks
        for i, c in enumerate(ranked_candidates):
            c["rank"] = i + 1
        
        # Build result
        result = {
            "status": "success",
            "tool": "screen_candidates",
            "total_input": len(candidates),
            "total_passed": len(ranked_candidates),
            "criteria_applied": {
                "pce_weight": pce_weight,
                "stability_weight": stability_weight,
                "synthesizability_weight": synthesizability_weight,
                "min_pce": min_pce,
                "max_energy_above_hull": max_energy_above_hull
            },
            "ranked_candidates": ranked_candidates[:5],  # Top 5
            "recommendation": ranked_candidates[0] if ranked_candidates else None
        }
        
        return result
    
    def _get_local_tool_schemas(self) -> list[dict]:
        """Get schemas for local tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "screen_candidates",
                    "description": "Screen and rank multiple candidate materials based on multi-criteria scoring. Use this after generate_material_structure to filter and select the best candidates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidates": {
                                "type": "array",
                                "description": "List of candidate materials from MatterGen output",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "formula": {"type": "string"},
                                        "predicted_pce": {"type": "number"},
                                        "energy_above_hull": {"type": "number"},
                                        "confidence": {"type": "number"}
                                    }
                                }
                            },
                            "criteria": {
                                "type": "object",
                                "description": "Screening criteria and weights",
                                "properties": {
                                    "pce_weight": {"type": "number", "description": "Weight for PCE score (default 0.4)"},
                                    "stability_weight": {"type": "number", "description": "Weight for stability (default 0.3)"},
                                    "synthesizability_weight": {"type": "number", "description": "Weight for synthesizability (default 0.3)"},
                                    "min_pce": {"type": "number", "description": "Minimum PCE threshold"},
                                    "max_energy_above_hull": {"type": "number", "description": "Maximum allowed instability"}
                                }
                            }
                        },
                        "required": ["candidates"]
                    }
                }
            }
        ]

    def set_tool_mode(self, mode: str) -> None:
        """
        Set the tool execution mode.
        
        Args:
            mode: "mock" or "interactive"
        """
        self._tool_mode = mode
        self._server_tools.set_mode(mode)
        self.logger.info(f"Tool mode set to: {mode}")

    def _get_system_prompt(
        self,
        state: dict[str, Any],
        default_prompt: str | None = None,
    ) -> str:
        """Return the domain-specific system prompt."""
        # Add mode information to prompt
        mode_note = f"\n\n**Current Tool Mode**: {self._tool_mode.upper()}"
        if self._tool_mode == "mock":
            mode_note += " (Using simulated results for testing)"
        else:
            mode_note += " (Tools will prompt for server results)"
        
        return SYSTEM_PROMPT + mode_note

    async def _get_tools_with_server_tools(self) -> list[dict[str, Any]]:
        """Get tool list including server tools and local tools"""
        # Get server tool schemas
        server_tool_schemas = self._server_tools.get_tool_schemas()
        
        # Get local tool schemas
        local_tool_schemas = self._get_local_tool_schemas()
        
        # Get MCP tools if available
        mcp_tools = []
        if self.registry.is_initialized():
            mcp_tools = await self.registry.get_tools_schema()
        
        # Merge all tools (local + server + MCP)
        all_tools = local_tool_schemas + server_tool_schemas + mcp_tools
        return all_tools

    async def _handle_tool_call(self, name: str, args: dict[str, Any]) -> str:
        """
        Handle tool call - route to local tools, server tools, or MCP tools.
        
        Args:
            name: Tool name
            args: Tool arguments
            
        Returns:
            Tool execution result as string
        """
        # Check if it's a local tool
        if name == "screen_candidates":
            candidates = args.get("candidates", [])
            criteria = args.get("criteria", {})
            result = self._screen_candidates(candidates, criteria)
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        # Check if it's a server tool
        if self._server_tools.has_tool(name):
            self.logger.info(f"Executing server tool: {name} (mode={self._tool_mode})")
            result_str = self._server_tools.execute(name, args)
            
            # If it's generate_material_structure, store candidates for later use
            if name == "generate_material_structure":
                try:
                    result_data = json.loads(result_str)
                    self._current_candidates = result_data.get("candidates", [])
                except:
                    pass
            
            return result_str
        
        # Otherwise, try MCP tool
        if self.registry.is_initialized():
            try:
                result = await self.registry.call_tool(name, args)
                return str(result) if result else "No result"
            except Exception as e:
                self.logger.error(f"MCP tool execution failed: {e}")
                return f"[ERROR] {e}"
        
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def autonomous_thinking(
        self,
        prompt: str,
        state: dict[str, Any],
        system_message: str | None = None,
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Execute ReAct thinking loop with server tool support.
        """
        if not self.llm:
            self.logger.error("LLM client not available")
            return {
                "response": "[ERROR] LLM not configured",
                "tool_calls": [],
                "tool_results": [],
                "iterations": 0,
            }

        # Get tools (including server tools)
        tools = await self._get_tools_with_server_tools()
        
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
        final_response_text = ""
        
        # Track if we've done any tool calls (need final synthesis)
        has_called_tools = False

        # ReAct loop
        while iterations < max_iterations:
            iterations += 1
            self.logger.debug(f"Thinking iteration {iterations}")

            response = await self.llm.ainvoke_streaming(
                messages, tools=tools if tools else None, print_stream=True,
            )

            if not self.llm.has_tool_calls(response):
                # No tool calls - this is the final response
                if response and hasattr(response, "content") and response.content:
                    final_response_text = response.content
                self.logger.debug("No tool calls, finishing")
                break

            tool_calls = self.llm.get_tool_calls(response)
            messages.append(response)

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                self.logger.info(f"Executing tool: {tool_name}")
                
                # Tool call visualization.
                is_server_tool = self._server_tools.has_tool(tool_name)
                is_local_tool = tool_name in self.LOCAL_TOOLS
                if is_local_tool:
                    tool_type = "🏠 Local"
                elif is_server_tool:
                    tool_type = "📍 Server"
                else:
                    tool_type = "🌐 MCP"
                
                print(f"\n🔧 [DesignAgent] Calling {tool_type} Tool: {tool_name}")
                print(f"   ⚙️  Mode: {self._tool_mode if is_server_tool else 'local'}")
                args_preview = json.dumps(tool_args, ensure_ascii=False)
                if len(args_preview) > 200:
                    args_preview = args_preview[:200] + '...'
                print(f"   📥 Arguments: {args_preview}")
                
                all_tool_calls.append(tc)
                has_called_tools = True

                try:
                    result_str = await self._handle_tool_call(tool_name, tool_args)
                    result_str = self._truncate_tool_output(result_str, tool_name)
                except Exception as e:
                    self.logger.error(f"Tool execution failed: {e}")
                    result_str = f"[ERROR] {e}"

                all_tool_results.append({
                    "tool": tool_name,
                    "result": result_str,
                })

                # Tool result visualization.
                try:
                    result_data = json.loads(result_str)
                    status = result_data.get("status", "unknown")
                    mode = result_data.get("_mode", "unknown")
                    print(f"   📤 Result: status={status}, mode={mode}")
                    
                    # Print key fields (debugging).
                    if tool_name == "check_synthesizability":
                        synth = result_data.get("synthesizable", result_data.get("result", "N/A"))
                        conf = result_data.get("confidence", "N/A")
                        print(f"      └─ synthesizable={synth}, confidence={conf}")
                    elif tool_name == "predict_synthesis_method":
                        method = result_data.get("method", "N/A")
                        conf = result_data.get("confidence", "N/A")
                        protocol = result_data.get("synthesis_protocol", "")
                        print(f"      └─ method={method}, confidence={conf}")
                        if protocol:
                            # Print full synthesis protocol (wrap every ~80 chars).
                            print(f"      ┌─ Synthesis Protocol:")
                            words = protocol.split()
                            line = "      │  "
                            for word in words:
                                if len(line) + len(word) > 90:
                                    print(line)
                                    line = "      │  " + word + " "
                                else:
                                    line += word + " "
                            if line.strip():
                                print(line)
                            print(f"      └──────────────────────────────────────────────────")
                    elif tool_name == "predict_precursors":
                        precursors = result_data.get("precursors", [])
                        prec_names = [p.get("name", p.get("formula", "?")) for p in precursors[:3]]
                        print(f"      └─ precursors={prec_names}{'...' if len(precursors) > 3 else ''}")
                    elif tool_name == "generate_material_structure":
                        candidates = result_data.get("candidates", [])
                        if candidates:
                            formulas = [c.get("formula", "?") for c in candidates[:3]]
                            print(f"      └─ candidates={formulas}{'...' if len(candidates) > 3 else ''}")
                except:
                    result_preview = result_str[:100] if len(result_str) > 100 else result_str
                    print(f"   📤 Result: {result_preview}{'...' if len(result_str) > 100 else ''}")

                tool_message = self.llm.create_tool_message(tool_id, result_str)
                messages.append(tool_message)
        
        # === CRITICAL: If we called tools but didn't get a final analysis, request one ===
        if has_called_tools and not final_response_text:
            self.logger.info("Requesting final synthesis after tool calls...")
            print(f"\n🔄 [DesignAgent] Synthesizing results from tool outputs...")
            
            # Add synthesis request
            synthesis_prompt = """Based on all the tool results above, please provide your final analysis and conclusions:

1. **Summary**: Summarize the key findings from the tool outputs
2. **Scientific Analysis**: Provide scientific interpretation of the results  
3. **Recommendations**: Give actionable recommendations based on the data
4. **Conclusion**: State the final conclusion for the task

Output your response as structured JSON with relevant fields based on the task type.
Do NOT call any more tools - just analyze the results you already have."""
            
            messages.append({"role": "user", "content": synthesis_prompt})
            
            # Get final synthesis (without tools to prevent more calls)
            synthesis_response = await self.llm.ainvoke(messages, tools=None)
            if synthesis_response and hasattr(synthesis_response, "content"):
                final_response_text = synthesis_response.content or ""
                print(f"   ✅ Final synthesis received ({len(final_response_text)} chars)")

        return {
            "response": final_response_text,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "iterations": iterations,
        }

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute experimental design based on MetaAgent's task.
        """
        print(f"\n{'='*70}")
        print(f"🛠️  [DesignAgent] Experimental Design Expert")
        print(f"{'='*70}")
        
        # Show available tools with categorization
        tools = await self._get_tools_with_server_tools()
        local_tool_names = list(self.LOCAL_TOOLS)
        server_tool_names = list(self.SERVER_TOOLS)
        mcp_tool_names = []
        for t in tools:
            name = t.get('function', {}).get('name', 'unknown')
            if name not in self.SERVER_TOOLS and name not in self.LOCAL_TOOLS:
                mcp_tool_names.append(name)
        
        print(f"\n📋 Tool Configuration:")
        print(f"   ├─ Mode: {self._tool_mode.upper()}")
        print(f"   ├─ 🏠 Local Tools ({len(local_tool_names)}): {local_tool_names}")
        print(f"   ├─ 📍 Server Tools ({len(server_tool_names)}): {server_tool_names}")
        print(f"   └─ 🌐 MCP Tools ({len(mcp_tool_names)}): {mcp_tool_names if mcp_tool_names else 'None'}")
        
        # Get context from state (includes upstream outputs)
        goal = state.get("goal", "")
        plan = state.get("plan", "")
        data_context = state.get("data_context", "")  # From DataAgent
        
        # Extract DesignAgent-specific task from MetaAgent's plan
        my_task = self._extract_my_task(plan, "DesignAgent")
        
        # Parse design requirements from goal/task
        requirements = self._parse_design_requirements(goal, my_task)
        
        # === Display upstream context clearly ===
        print(f"\n📊 Upstream Context:")
        print(f"   ├─ 🎯 Goal: {(goal or '')[:80]}{'...' if len(goal or '') > 80 else ''}")
        print(f"   ├─ 📝 Task: {my_task}")
        print(f"   └─ 📚 Data (DataAgent): {len(data_context)} chars")
        
        print(f"\n📐 Design Requirements:")
        for key, value in requirements.items():
            if value is not None:
                print(f"   ├─ {key}: {value}")
        
        if self._tool_mode == "interactive":
            print(f"\n⚠️  INTERACTIVE MODE:")
            print(f"   When server tools are called, you'll need to:")
            print(f"   1. Run the displayed command on your server")
            print(f"   2. Paste the result back into this terminal")
            print(f"   3. Type 'END' on a new line when done")
        
        print(f"\n{'─'*70}")
        print(f"🚀 Starting design process...")
        print(f"{'─'*70}\n")

        # Build design prompt - flexible, not rigid workflow
        prompt = f"""# 🧠 EXPERIMENTAL DESIGN MISSION

## Strategic Goal
{goal}

## Specific Task from MetaAgent
{my_task}

## Parsed Design Requirements
{json.dumps(requirements, indent=2, ensure_ascii=False)}

## Knowledge Context (from DataAgent)
{safe_truncate(data_context, 5000, default="No specific literature data provided. Use fundamental chemical principles.")}

# 🎯 YOUR MISSION
Based on the task above, use the appropriate tools to complete the design:
- Design new materials → `generate_material_structure`, optionally `screen_candidates`
- Check synthesizability → `check_synthesizability`
- Determine synthesis method → `predict_synthesis_method`
- Find precursors → `predict_precursors` (may need method first)

# 📋 OUTPUT REQUIREMENTS
After completing tool calls, provide your results as JSON. Include:
- The key results from tools you called (formula, synthesizability, method, precursors, etc.)
- Your analysis and scientific reasoning
- Status of the task (success/partial/failed)

Only include fields relevant to what you actually did. Don't include empty sections.
"""

        result = await self.autonomous_thinking(
            prompt=prompt,
            state=state,
            system_message=SYSTEM_PROMPT,
            max_iterations=10,
        )

        response_text = result.get("response", "")
        tool_results = result.get("tool_results", [])
        
        # Parse parameters from LLM response
        experimental_params = self._parse_parameters(response_text)
        
        # === CRITICAL: Merge tool results into experimental_params ===
        # This ensures tool outputs are captured even if LLM doesn't summarize them well
        experimental_params = self._merge_tool_results(experimental_params, tool_results)
        
        # Add metadata
        experimental_params["_tool_mode"] = self._tool_mode
        experimental_params["_tool_calls"] = len(result.get("tool_calls", []))
        experimental_params["_iterations"] = result.get("iterations", 0)
        
        # Ensure standard output fields exist
        if "status" not in experimental_params:
            experimental_params["status"] = "success" if experimental_params.get("composition") else "partial"
        
        # === Display output summary ===
        print(f"\n{'─'*70}")
        print(f"✅ [DesignAgent] Design Complete")
        print(f"   ├─ Task Type: {experimental_params.get('task_type', 'N/A')}")
        print(f"   ├─ Tool Calls: {experimental_params['_tool_calls']}")
        print(f"   ├─ Status: {experimental_params.get('status', 'N/A')}")
        
        if "composition" in experimental_params and experimental_params["composition"]:
            comp = experimental_params["composition"]
            print(f"   ├─ Formula: {comp.get('formula', 'N/A')}")
            synth = comp.get('synthesizability', {})
            if synth:
                print(f"   ├─ Synthesizable: {synth.get('result', 'N/A')}")
        
        proc = experimental_params.get("process", {})
        if proc:
            print(f"   ├─ Method: {proc.get('method', 'N/A')}")
            protocol = proc.get('synthesis_protocol', '')
            if protocol:
                print(f"   └─ Protocol: {protocol[:120]}{'...' if len(protocol) > 120 else ''}")
            else:
                print(f"   └─ Protocol: N/A")
        else:
            print(f"   └─ Process: Not specified")

        return {"experimental_params": experimental_params}
    
    def _parse_design_requirements(self, goal: str, task: str) -> dict:
        """Parse design requirements from goal and task text."""
        requirements = {
            "target_pce": None,
            "target_voc": None,
            "target_jsc": None,
            "target_ff": None,
            "target_bandgap": None,
            "stability_requirement": None,
            "composition_constraints": None,
            "special_requirements": []
        }
        
        combined = f"{goal} {task}".lower()
        
        # Extract PCE target
        pce_match = re.search(r'pce[^\d]*(\d+(?:\.\d+)?)\s*%?', combined, re.I)
        if pce_match:
            requirements["target_pce"] = float(pce_match.group(1))
        elif "high efficiency" in combined:
            requirements["target_pce"] = 25.0
        
        # Extract Voc target
        voc_match = re.search(r'voc[^\d]*(\d+(?:\.\d+)?)\s*v', combined, re.I)
        if voc_match:
            requirements["target_voc"] = float(voc_match.group(1))
        elif "open-circuit voltage" in combined or "open circuit voltage" in combined:
            voc_match2 = re.search(r'(\d+(?:\.\d+)?)\s*v', combined, re.I)
            if voc_match2:
                requirements["target_voc"] = float(voc_match2.group(1))
        
        # Extract bandgap target
        bg_match = re.search(r'band\s*gap[^\d]*(\d+(?:\.\d+)?)[^\d]*(\d+(?:\.\d+)?)?\s*ev', combined, re.I)
        if bg_match:
            requirements["target_bandgap"] = float(bg_match.group(1))
        elif "wide bandgap" in combined or "wide-bandgap" in combined:
            requirements["target_bandgap"] = 1.75
        elif "narrow bandgap" in combined or "narrow-bandgap" in combined:
            requirements["target_bandgap"] = 1.3
        
        # Detect composition constraints
        if "lead-free" in combined:
            requirements["composition_constraints"] = "lead-free"
            requirements["special_requirements"].append("lead-free perovskite")
        if "sn-based" in combined:
            requirements["composition_constraints"] = "Sn-based"
            requirements["special_requirements"].append("Sn-based perovskite")
        if "cs-doped" in combined or "cs doped" in combined:
            requirements["special_requirements"].append("Cs-doped")
        
        # Detect stability requirements
        if "stability" in combined or "stable" in combined:
            requirements["stability_requirement"] = "high"
        
        # Remove None values for cleaner output
        return {k: v for k, v in requirements.items() if v is not None and v != []}
    
    def _extract_my_task(self, plan: str | dict, agent_name: str) -> str:
        """Extract specific task for this agent from MetaAgent's plan."""
        if not plan:
            return "Design optimal perovskite material and process"
        
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
                    return str(plan)[:500]
            except (json.JSONDecodeError, KeyError):
                if agent_name in str(plan):
                    for line in str(plan).split('\n'):
                        if agent_name in line:
                            return line.replace(agent_name, "").strip(': -')
                return str(plan)[:500]
        
        # Extract agent-specific task
        agent_tasks = plan_data.get("agent_tasks", {})
        task = agent_tasks.get(agent_name, "")
        
        if task and str(task).upper() != "SKIP":
            return task
        else:
            return plan_data.get("iteration_focus", "Design optimal material and process")

    def _merge_tool_results(self, params: dict[str, Any], tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Merge tool call results into experimental_params.
        This ensures we capture tool outputs even if LLM didn't summarize them properly.
        """
        if not tool_results:
            return params
        
        # Ensure composition dict exists
        if "composition" not in params or not params["composition"]:
            params["composition"] = {}
        
        # Ensure process dict exists
        if "process" not in params or not params["process"]:
            params["process"] = {}
        
        for tr in tool_results:
            tool_name = tr.get("tool", "")
            result_str = tr.get("result", "")
            
            # Skip if empty or error
            if not result_str or result_str.startswith("[ERROR]"):
                continue
            
            # Parse JSON result
            try:
                result_data = json.loads(result_str)
            except (json.JSONDecodeError, TypeError):
                continue
            
            # Skip if status is error
            if result_data.get("status") == "error":
                continue
            
            # === Merge based on tool type ===
            if tool_name == "generate_material_structure":
                candidates = result_data.get("candidates", [])
                if candidates:
                    # Take the first/best candidate
                    best = candidates[0]
                    params["composition"]["formula"] = best.get("formula", params["composition"].get("formula"))
                    params["composition"]["structure_type"] = best.get("structure_type", "3D")
                    params["composition"]["predicted_pce"] = best.get("predicted_pce")
                    params["composition"]["predicted_bandgap"] = best.get("predicted_bandgap")
                    params["composition"]["energy_above_hull"] = best.get("energy_above_hull")
                    params["composition"]["all_candidates"] = candidates
                    
            elif tool_name == "check_synthesizability":
                synth_result = result_data.get("synthesizability", result_data.get("result", {}))
                if isinstance(synth_result, dict):
                    params["composition"]["synthesizability"] = synth_result
                else:
                    params["composition"]["synthesizability"] = {
                        "result": synth_result,
                        "raw": result_data
                    }
                # Also capture formula if provided
                if "formula" in result_data and not params["composition"].get("formula"):
                    params["composition"]["formula"] = result_data["formula"]
                    
            elif tool_name == "predict_synthesis_method":
                method = result_data.get("method", result_data.get("synthesis_method", {}))
                if isinstance(method, dict):
                    params["process"]["method"] = method.get("name", method.get("type", "unknown"))
                    params["process"]["method_details"] = method
                elif isinstance(method, str):
                    params["process"]["method"] = method
                # Capture full synthesis protocol text.
                if "synthesis_protocol" in result_data:
                    params["process"]["synthesis_protocol"] = result_data["synthesis_protocol"]
                params["process"]["method_raw"] = result_data
                    
            elif tool_name == "predict_precursors":
                precursors = result_data.get("precursors", [])
                params["process"]["precursors"] = precursors
                params["process"]["precursors_raw"] = result_data
                
            elif tool_name == "screen_candidates":
                params["screening_results"] = result_data
        
        # Set status based on what we have
        has_formula = bool(params.get("composition", {}).get("formula"))
        has_method = bool(params.get("process", {}).get("method"))
        has_precursors = bool(params.get("process", {}).get("precursors"))
        
        if has_formula and has_method and has_precursors:
            params["status"] = "success"
        elif has_formula or has_method:
            params["status"] = "partial"
        else:
            params["status"] = "incomplete"
        
        return params

    def _parse_parameters(self, response: str) -> dict[str, Any]:
        """
        Robustly parse JSON from the response.
        """
        try:
            # First try to match Markdown code block
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if match:
                return json.loads(match.group(1))

            # Then try to find content surrounded by curly braces
            # Find the first { and last }
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                json_str = response[start : end + 1]
                return json.loads(json_str)

        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON parameters from response")
        
        # Return minimal structure if parsing fails
        return {
            "design_rationale": safe_truncate(response, 500, default="No response"),
            "parse_error": True
        }


# =============================================================================
# Factory function for easy agent creation
# =============================================================================

def create_design_agent(
    settings: Settings | None = None,
    mode: str = "mock"
) -> DesignAgent:
    """
    Create a DesignAgent with specified mode.
    
    Args:
        settings: Configuration settings
        mode: "mock" for testing, "interactive" for server input
        
    Returns:
        Configured DesignAgent instance
    """
    return DesignAgent(settings=settings, tool_mode=mode)