"""
MemoryAgent - Strategic Knowledge Keeper for PSC_Agents.

Responsible for:
1. Validating and Archiving the current iteration's key insights.
2. Extracting and preserving COMPLETE experimental parameters (formula, method, protocol, precursors).
3. Updating the long-term memory log with goal-aligned summaries.
4. Passing the FULL state (including recent Analysis) back to MetaAgent for the next planning phase.

Author: PSC_Agents Team
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.base_agent import BaseAgent
from core.config import Settings


# === System Prompt ===
SYSTEM_PROMPT = """You are MemoryAgent - Strategic Knowledge Keeper of PSC_Agents.

## Core Mission
Your goal is to distill the dynamic workflow state into high-value "Knowledge Capsules". You are the filter that separates critical scientific insights from experimental noise, ensuring that the MetaAgent can learn from both brilliant successes and instructive failures.

## Your Analytical Role
1. **Insight Distillation**: Do not just record results. Identify the *why* (e.g., "The high Voc is likely due to the cation-mixing effect identified in Analysis").
2. **Knowledge Compression**: Keep the most important experimental details - formula, synthesis protocol, precursors, key parameters.
3. **Trend Detection**: Evaluate how the current result compares to the long-term goal. Is the system converging toward the target PCE/Stability?
4. **Goal Alignment**: Always evaluate results against the original research goal.

## Archival Principles
- **Success Case**: Archive the precise recipe (formula + full synthesis protocol + precursors) as a "Golden Template".
- **Failure Case**: Focus on the "Root Cause Diagnosis" from Analysis. What should the team NEVER do again?
- **Complete Recipe**: Always preserve the complete synthesis method, not just the name.
- **Scientific Integrity**: Ensure all metrics (PCE, T80, Voc) are accurately preserved with their respective units.

## Output Requirement
Return a structured JSON Knowledge Capsule that acts as the "Scientific Lab Notebook" for the next research iteration.
"""


class MemoryAgent(BaseAgent):
    """
    Strategic Memory Agent.
    
    It selectively archives key insights into the memory_log but PRESERVES 
    the current state so the MetaAgent can see the full details of the 
    just-completed iteration.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(name="MemoryAgent", settings=settings)

    def _get_system_prompt(
        self,
        state: dict[str, Any],
        default_prompt: str | None = None,
    ) -> str:
        return SYSTEM_PROMPT

    def _extract_literature_references(self, data_context: str) -> list[dict[str, Any]]:
        """Extract literature references from DataAgent's data_context."""
        refs = []
        if not data_context:
            return refs
        
        try:
            # Try to parse as JSON
            data = json.loads(data_context)
            extracted_data = data.get("extracted_data", [])
            
            for paper in extracted_data:
                ref = {
                    "paper_id": paper.get("arxiv_id", paper.get("paper_id", "Unknown")),
                    "title": paper.get("title", "Unknown"),
                    "key_findings": paper.get("key_findings", []),
                    "performance_metrics": paper.get("performance_metrics", {}),
                    "materials": paper.get("materials", {}),
                }
                refs.append(ref)
                
        except (json.JSONDecodeError, TypeError):
            # If not JSON, try to extract paper IDs using regex
            import re
            arxiv_ids = re.findall(r'\d{4}\.\d{4,5}(?:v\d+)?', data_context)
            for arxiv_id in set(arxiv_ids):
                refs.append({"paper_id": arxiv_id, "title": "Unknown", "key_findings": []})
        
        return refs

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute selective memory logic with FULL state visibility.
        """
        print(f"\n{'='*60}")
        print(f"💾 [MemoryAgent] Archiving iteration...")
        print(f"{'='*60}")
        
        # Show available tools (MemoryAgent is pure reasoning, no tools)
        print(f"\n🛠️  Available Tools Summary:")
        print(f"   📍 Local Tools (0): None (Pure reasoning agent)")
        print(f"   🌐 MCP Tools (0): None")
        print(f"   📊 Total: 0 tools")
        
        # 1. Get ALL context from state (强制对齐)
        current_iteration = state.get("current_iteration", 0)
        goal = state.get("goal", "")
        plan = state.get("plan", "")
        
        # === CRITICAL: Extract ALL upstream outputs ===
        data_context = state.get("data_context", "")  # From DataAgent
        experimental_params = state.get("experimental_params", {})  # From DesignAgent
        fab_results = state.get("fab_results", {})  # From FabAgent
        analysis_report = state.get("analysis_report", "")  # From AnalysisAgent
        
        # Print full state visibility
        print(f"📋 Iteration: {current_iteration}")
        print(f"📚 Data context: {len(data_context)} chars")
        print(f"🧪 Experimental params: {list(experimental_params.keys()) if experimental_params else 'None'}")
        print(f"🏭 Fab results: {'Available' if fab_results else 'None'}")
        print(f"📊 Analysis report: {len(analysis_report)} chars")
        
        # === Extract COMPLETE experimental details ===
        formula = "Unknown"
        method = "Unknown"
        synthesis_protocol = ""
        precursors = []
        predicted_pce = "N/A"
        predicted_voc = "N/A"
        predicted_jsc = "N/A"
        predicted_ff = "N/A"
        predicted_bandgap = "N/A"
        predicted_t80 = "N/A"
        
        if experimental_params:
            # Composition info
            comp = experimental_params.get("composition", {})
            formula = comp.get("formula", experimental_params.get("formula", "Unknown"))
            
            # Process info - CRITICAL: Extract full synthesis details
            process = experimental_params.get("process", {})
            method = process.get("method", "Unknown")
            synthesis_protocol = process.get("synthesis_protocol", "")
            precursors = process.get("precursors", [])
            
            # Format precursors for display
            if precursors and isinstance(precursors, list):
                precursor_names = [p.get("name", p.get("formula", str(p))) for p in precursors if isinstance(p, dict)]
                precursors_str = ", ".join(precursor_names) if precursor_names else str(precursors)
            else:
                precursors_str = str(precursors) if precursors else "Not specified"
        
        if fab_results:
            metrics = fab_results.get("predicted_metrics", fab_results.get("metrics", fab_results))
            if isinstance(metrics, dict):
                predicted_pce = metrics.get("PCE_percent", metrics.get("pce", "N/A"))
                predicted_voc = metrics.get("Voc_V", metrics.get("voc", "N/A"))
                predicted_jsc = metrics.get("Jsc_mAcm2", metrics.get("jsc", "N/A"))
                predicted_ff = metrics.get("FF_percent", metrics.get("ff", "N/A"))
                predicted_bandgap = metrics.get("bandgap_eV", metrics.get("band_gap", "N/A"))
                predicted_t80 = metrics.get("T80_hours", "N/A")

        # Print extracted core info
        print(f"\n📋 Core Information Extracted:")
        print(f"   Formula: {formula}")
        print(f"   Method: {method}")
        print(f"   Protocol: {synthesis_protocol[:100]}{'...' if len(synthesis_protocol) > 100 else ''}" if synthesis_protocol else "   Protocol: N/A")
        print(f"   Precursors: {precursors_str if 'precursors_str' in dir() else 'N/A'}")
        print(f"   Predicted PCE: {predicted_pce}")

        # 2. Build archival prompt with FULL context and GOAL alignment
        prompt = f"""# 💾 ARCHIVAL MISSION: ITERATION {current_iteration}

## 🎯 ORIGINAL RESEARCH GOAL (Critical - All analysis must align with this!)
{goal}

## 📋 MetaAgent's Plan for this Iteration
{plan if isinstance(plan, str) else json.dumps(plan, indent=2, ensure_ascii=False)}

# 🔍 COMPLETE EVIDENCE CHAIN (All Agent Outputs)

### [1] DataAgent Literature Context:
{data_context[:2000] + '...' if len(data_context) > 2000 else data_context if data_context else 'No literature data collected'}

### [2] DesignAgent Recipe (COMPLETE):
**Formula**: {formula}
**Method**: {method}
**Synthesis Protocol**: 
{synthesis_protocol if synthesis_protocol else 'Not provided'}

**Precursors**: {precursors_str if 'precursors_str' in dir() else 'Not specified'}

**Full Parameters**:
```json
{json.dumps(experimental_params, indent=2, ensure_ascii=False)}
```

### [3] FabAgent Predictions:
- PCE: {predicted_pce}
- Voc: {predicted_voc}
- Jsc: {predicted_jsc}
- FF: {predicted_ff}
- Bandgap: {predicted_bandgap}
```json
{json.dumps(fab_results, indent=2, ensure_ascii=False) if fab_results else 'No predictions available'}
```

### [4] AnalysisAgent Diagnosis:
{analysis_report if analysis_report else 'No analysis available'}

# 🧠 YOUR TASK: Create Comprehensive Knowledge Capsule

**IMPORTANT**: Evaluate everything against the ORIGINAL GOAL: "{goal[:200]}..."

You MUST extract and archive:
1. **Complete Recipe**: Formula + Method + Full Synthesis Protocol + Precursors
2. **Performance Metrics**: PCE, Voc, Jsc, FF, Bandgap
3. **Goal Alignment**: Does this iteration's result address the original goal?
4. **Success/Failure Analysis**: Why did it work or not work?
5. **Actionable Learning**: What should the next iteration do differently?

# 🏁 OUTPUT FORMAT (JSON)
```json
{{
  "iteration_id": {current_iteration},
  "goal_summary": "<1-sentence summary of original goal>",
  "recipe": {{
    "formula": "{formula}",
    "method": "{method}",
    "synthesis_protocol": "<FULL protocol - copy it entirely if available>",
    "precursors": "<list of precursors>",
    "key_parameters": "<annealing temp, spin speed, etc.>"
  }},
  "predictions": {{
    "pce": "{predicted_pce}",
    "voc": "{predicted_voc}",
    "bandgap": "{predicted_bandgap}"
  }},
  "goal_alignment": {{
    "aligned": true/false,
    "reason": "<why is this aligned or not aligned with the goal?>"
  }},
  "verdict": "SUCCESS/FAILURE/PARTIAL",
  "root_cause": "<if failure, what went wrong?>",
  "critical_learning": "<most important lesson>",
  "next_iteration_advice": "<specific advice for MetaAgent>"
}}
```
"""

        result = await self.autonomous_thinking(
            prompt=prompt,
            state=state,
            system_message=SYSTEM_PROMPT,
            max_iterations=2,
        )

        response_text = result.get("response", "")
        
        # 3. Parse and build structured log entry
        entry_json = self._extract_json_block(response_text)
        
        if entry_json:
            recipe = entry_json.get("recipe", {})
            predictions = entry_json.get("predictions", {})
            goal_alignment = entry_json.get("goal_alignment", {})
            
            # Format as comprehensive Markdown for MetaAgent
            iteration_entry = (
                f"### Iteration {current_iteration} [{entry_json.get('verdict', 'N/A')}]\n"
                f"**Goal Alignment**: {'✅ Yes' if goal_alignment.get('aligned') else '❌ No'} - {goal_alignment.get('reason', 'N/A')}\n\n"
                f"**Recipe**:\n"
                f"- Formula: {recipe.get('formula', formula)}\n"
                f"- Method: {recipe.get('method', method)}\n"
                f"- Protocol: {recipe.get('synthesis_protocol', synthesis_protocol)[:300]}{'...' if len(recipe.get('synthesis_protocol', synthesis_protocol)) > 300 else ''}\n"
                f"- Precursors: {recipe.get('precursors', 'N/A')}\n"
                f"- Key Params: {recipe.get('key_parameters', 'N/A')}\n\n"
                f"**Predictions**: PCE={predictions.get('pce', predicted_pce)}, Voc={predictions.get('voc', predicted_voc)}, Eg={predictions.get('bandgap', predicted_bandgap)}\n\n"
                f"**Root Cause**: {entry_json.get('root_cause', 'N/A')}\n"
                f"**Learning**: {entry_json.get('critical_learning', 'N/A')}\n"
                f"**Next Advice**: {entry_json.get('next_iteration_advice', 'N/A')}"
            )
            
            # Also store structured data for potential retrieval
            # Parse data_context for literature references
            literature_refs = self._extract_literature_references(data_context)
            
            structured_memory = {
                "iteration": current_iteration,
                "goal_summary": entry_json.get("goal_summary", ""),
                "formula": recipe.get("formula", formula),
                "method": recipe.get("method", method),
                "synthesis_protocol": recipe.get("synthesis_protocol", synthesis_protocol),
                "precursors": recipe.get("precursors", ""),
                "pce": predictions.get("pce", predicted_pce),
                "verdict": entry_json.get("verdict", "UNKNOWN"),
                "aligned_with_goal": goal_alignment.get("aligned", False),
                "learning": entry_json.get("critical_learning", ""),
                "advice": entry_json.get("next_iteration_advice", ""),
                # IMPORTANT: Include literature evidence
                "literature_refs": literature_refs,
                "data_context_summary": data_context[:1500] if data_context else ""
            }
        else:
            # Fallback: create entry from known values
            iteration_entry = (
                f"### Iteration {current_iteration} [PARSE_FAILED]\n"
                f"**Formula**: {formula}\n"
                f"**Method**: {method}\n"
                f"**Protocol**: {synthesis_protocol[:200] if synthesis_protocol else 'N/A'}\n"
                f"**PCE**: {predicted_pce}\n"
                f"**Raw Response**: {response_text[:500]}"
            )
            # Parse data_context for literature references (fallback)
            literature_refs = self._extract_literature_references(data_context)
            
            structured_memory = {
                "iteration": current_iteration,
                "formula": formula,
                "method": method,
                "synthesis_protocol": synthesis_protocol,
                "pce": predicted_pce,
                "literature_refs": literature_refs,
                "data_context_summary": data_context[:1500] if data_context else ""
            }

        self.logger.info(f"Archived Iteration {current_iteration}. Formula: {formula}, PCE: {predicted_pce}")
        
        # Print memory output
        print(f"\n📝 [MemoryAgent] Knowledge Capsule:")
        print(f"{'-'*40}")
        print(f"{iteration_entry}")
        print(f"{'-'*40}")
        print(f"\n🔄 Feedback to MetaAgent:")
        print(f"  Next iteration: {current_iteration + 1}")
        print(f"  Memory log updated: +1 entry")
        print(f"  Goal aligned: {structured_memory.get('aligned_with_goal', 'Unknown')}")

        # 4. Return Memory Log update and Iteration count
        # Keep analysis_report for MetaAgent's next planning phase
        return {
            "memory_log": [iteration_entry],  # Append to history
            "structured_memory": [structured_memory],  # Structured data for retrieval
            "current_iteration": current_iteration + 1,
        }

    def _extract_json_block(self, text: str) -> Any | None:
        """Robustly extract JSON from Markdown."""
        try:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return json.loads(match.group(1))
            
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
        except Exception:
            pass
        return None