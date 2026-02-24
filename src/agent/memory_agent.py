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


# === Helper Functions for Type Safety ===
def safe_str(value: Any, default: str = "") -> str:
    """Safely convert any value to string, handling None, list, dict, etc."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '; '.join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def safe_truncate(value: Any, max_len: int, suffix: str = "...", default: str = "N/A") -> str:
    """Safely truncate any value to string with max length."""
    text = safe_str(value, default)
    if len(text) > max_len:
        return text[:max_len] + suffix
    return text


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
        """Extract COMPACT literature references from DataAgent's data_context.
        Only keeps essential info to avoid context explosion."""
        refs = []
        if not data_context:
            return refs
        
        try:
            # Try to parse as JSON
            data = json.loads(data_context)
            extracted_data = data.get("extracted_data", [])
            
            for paper in extracted_data:
                if not isinstance(paper, dict):
                    continue
                    
                # Extract only essential fields with length limits
                key_findings = paper.get("key_findings", {})
                # Convert key_findings to compact string if it's a dict
                if isinstance(key_findings, dict):
                    findings_str = "; ".join(f"{k}:{v}" for k, v in list(key_findings.items())[:5])
                elif isinstance(key_findings, list):
                    findings_str = "; ".join(safe_str(f)[:100] for f in key_findings[:3])
                else:
                    findings_str = safe_str(key_findings)[:300]
                
                # Compact performance metrics
                metrics = paper.get("performance_metrics", {})
                metrics_compact = {}
                if isinstance(metrics, dict):
                    for k in ["PCE", "Voc", "Jsc", "FF"]:
                        if metrics.get(k) is not None:
                            metrics_compact[k] = metrics[k]
                
                # Compact materials info
                materials = paper.get("materials", {})
                composition_raw = materials.get("composition", "") if isinstance(materials, dict) else ""
                composition = safe_str(composition_raw)
                
                ref = {
                    "paper_id": safe_str(paper.get("arxiv_id") or paper.get("paper_id"), "Unknown"),
                    "title": safe_truncate(paper.get("title"), 100, "...", "Unknown"),
                    "findings": safe_truncate(findings_str, 300),
                    "metrics": metrics_compact,  # Only non-null metrics
                    "composition": safe_truncate(composition, 100, "", ""),
                }
                refs.append(ref)
                
        except (json.JSONDecodeError, TypeError, AttributeError):
            # If not JSON, try to extract paper IDs using regex
            if isinstance(data_context, str):
                arxiv_ids = re.findall(r'\d{4}\.\d{4,5}(?:v\d+)?', data_context)
                for arxiv_id in set(arxiv_ids):
                    refs.append({"paper_id": arxiv_id, "title": "Unknown", "findings": ""})
        
        return refs

    def _format_literature_summary(self, literature_refs: list[dict]) -> str:
        """Format literature references into a human-readable summary."""
        if not literature_refs:
            return "No literature data collected this iteration."
        
        lines = [f"📚 Analyzed {len(literature_refs)} papers:"]
        for i, ref in enumerate(literature_refs[:10], 1):  # Max 10 papers
            if not isinstance(ref, dict):
                continue
            paper_id = safe_str(ref.get("paper_id"), "?")
            title = safe_truncate(ref.get("title"), 60, "...", "Unknown")
            findings = safe_str(ref.get("findings"), "")
            metrics = ref.get("metrics") if isinstance(ref.get("metrics"), dict) else {}
            composition = safe_str(ref.get("composition"), "")
            
            line = f"[{paper_id}] {title}"
            if composition:
                line += f" | Comp: {composition}"
            if metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                line += f" | {metrics_str}"
            if findings:
                line += f" | Findings: {findings[:80]}..."
            
            lines.append(f"  {i}. {line}")
        
        if len(literature_refs) > 10:
            lines.append(f"  ... and {len(literature_refs) - 10} more papers")
        
        return "\n".join(lines)

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
        
        # 1) Get ALL context from state (forced alignment).
        current_iteration = state.get("current_iteration", 0)
        goal = state.get("goal", "")
        plan = state.get("plan", "")
        
        # === CRITICAL: Extract ALL upstream outputs ===
        data_context = state.get("data_context", "")  # From DataAgent
        experimental_params = state.get("experimental_params", {})  # From DesignAgent
        fab_results = state.get("fab_results", {})  # From FabAgent
        analysis_report = state.get("analysis_report", "")  # From AnalysisAgent
        
        # Ensure string types for safety
        data_context = safe_str(data_context)
        analysis_report = safe_str(analysis_report)
        
        # Print full state visibility
        print(f"📋 Iteration: {current_iteration}")
        print(f"📚 Data context: {len(data_context)} chars")
        print(f"🧪 Experimental params: {list(experimental_params.keys()) if isinstance(experimental_params, dict) and experimental_params else 'None'}")
        print(f"🏭 Fab results: {'Available' if fab_results else 'None'}")
        print(f"📊 Analysis report: {len(analysis_report)} chars")
        
        # === Extract COMPLETE experimental details ===
        formula = "Unknown"
        method = "Unknown"
        synthesis_protocol = ""
        precursors = []
        precursors_str = "Not specified"
        predicted_pce = "N/A"
        predicted_voc = "N/A"
        predicted_jsc = "N/A"
        predicted_ff = "N/A"
        predicted_bandgap = "N/A"
        predicted_t80 = "N/A"
        
        if experimental_params and isinstance(experimental_params, dict):
            # Composition info
            comp = experimental_params.get("composition", {})
            if isinstance(comp, dict):
                formula = safe_str(comp.get("formula") or experimental_params.get("formula"), "Unknown")
            else:
                formula = safe_str(experimental_params.get("formula"), "Unknown")
            
            # Process info - CRITICAL: Extract full synthesis details
            process = experimental_params.get("process", {})
            if isinstance(process, dict):
                method = safe_str(process.get("method"), "Unknown")
                # Handle synthesis_protocol which might be a list or string
                synthesis_protocol = safe_str(process.get("synthesis_protocol"), "")
                precursors = process.get("precursors", [])
            
            # Format precursors for display
            if precursors and isinstance(precursors, list):
                precursor_names = []
                for p in precursors:
                    if isinstance(p, dict):
                        precursor_names.append(safe_str(p.get("name") or p.get("formula"), str(p)))
                    else:
                        precursor_names.append(safe_str(p))
                precursors_str = ", ".join(precursor_names) if precursor_names else safe_str(precursors)
            elif precursors:
                precursors_str = safe_str(precursors)
        
        if fab_results and isinstance(fab_results, dict):
            metrics = fab_results.get("predicted_metrics") or fab_results.get("metrics") or fab_results
            if isinstance(metrics, dict):
                predicted_pce = safe_str(metrics.get("PCE_percent") or metrics.get("pce"), "N/A")
                predicted_voc = safe_str(metrics.get("Voc_V") or metrics.get("voc"), "N/A")
                predicted_jsc = safe_str(metrics.get("Jsc_mAcm2") or metrics.get("jsc"), "N/A")
                predicted_ff = safe_str(metrics.get("FF_percent") or metrics.get("ff"), "N/A")
                predicted_bandgap = safe_str(metrics.get("bandgap_eV") or metrics.get("band_gap"), "N/A")
                predicted_t80 = safe_str(metrics.get("T80_hours"), "N/A")

        # Print extracted core info
        print(f"\n📋 Core Information Extracted:")
        print(f"   Formula: {formula}")
        print(f"   Method: {method}")
        print(f"   Protocol: {safe_truncate(synthesis_protocol, 100)}" if synthesis_protocol else "   Protocol: N/A")
        print(f"   Precursors: {precursors_str}")
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

**IMPORTANT**: Evaluate everything against the ORIGINAL GOAL: "{(goal or '')[:200]}..."

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

        response_text = safe_str(result.get("response"), "")
        
        # 3. Parse and build structured log entry
        entry_json = self._extract_json_block(response_text)
        
        if entry_json and isinstance(entry_json, dict):
            recipe = entry_json.get("recipe", {}) if isinstance(entry_json.get("recipe"), dict) else {}
            predictions = entry_json.get("predictions", {}) if isinstance(entry_json.get("predictions"), dict) else {}
            goal_alignment = entry_json.get("goal_alignment", {}) if isinstance(entry_json.get("goal_alignment"), dict) else {}
            
            # Format as comprehensive Markdown for MetaAgent
            # Use safe_str to handle None, list, dict, etc.
            protocol_raw = recipe.get('synthesis_protocol') or synthesis_protocol or 'N/A'
            protocol_text = safe_str(protocol_raw)
            protocol_display = safe_truncate(protocol_text, 300)
            
            # Safe extraction of all fields
            recipe_formula = safe_str(recipe.get('formula') or formula)
            recipe_method = safe_str(recipe.get('method') or method)
            recipe_precursors = safe_str(recipe.get('precursors'), 'N/A')
            recipe_key_params = safe_str(recipe.get('key_parameters'), 'N/A')
            pred_pce = safe_str(predictions.get('pce') or predicted_pce)
            pred_voc = safe_str(predictions.get('voc') or predicted_voc)
            pred_bandgap = safe_str(predictions.get('bandgap') or predicted_bandgap)
            verdict = safe_str(entry_json.get('verdict'), 'N/A')
            root_cause = safe_str(entry_json.get('root_cause'), 'N/A')
            learning = safe_str(entry_json.get('critical_learning'), 'N/A')
            advice = safe_str(entry_json.get('next_iteration_advice'), 'N/A')
            align_reason = safe_str(goal_alignment.get('reason'), 'N/A')
            
            iteration_entry = (
                f"### Iteration {current_iteration} [{verdict}]\n"
                f"**Goal Alignment**: {'✅ Yes' if goal_alignment.get('aligned') else '❌ No'} - {align_reason}\n\n"
                f"**Recipe**:\n"
                f"- Formula: {recipe_formula}\n"
                f"- Method: {recipe_method}\n"
                f"- Protocol: {protocol_display}\n"
                f"- Precursors: {recipe_precursors}\n"
                f"- Key Params: {recipe_key_params}\n\n"
                f"**Predictions**: PCE={pred_pce}, Voc={pred_voc}, Eg={pred_bandgap}\n\n"
                f"**Root Cause**: {root_cause}\n"
                f"**Learning**: {learning}\n"
                f"**Next Advice**: {advice}"
            )
            
            # Also store structured data for potential retrieval
            # Parse data_context for literature references
            literature_refs = self._extract_literature_references(data_context)
            
            # Build a COMPACT summary of literature findings (not raw JSON)
            lit_summary = self._format_literature_summary(literature_refs)
            
            structured_memory = {
                "iteration": current_iteration,
                "goal_summary": safe_str(entry_json.get("goal_summary"), ""),
                "formula": recipe_formula,
                "method": recipe_method,
                "synthesis_protocol": safe_str(recipe.get("synthesis_protocol") or synthesis_protocol),
                "precursors": recipe_precursors,
                "pce": pred_pce,
                "verdict": verdict,
                "aligned_with_goal": bool(goal_alignment.get("aligned")),
                "learning": learning,
                "advice": advice,
                # IMPORTANT: Include literature evidence (compact)
                "literature_refs": literature_refs,
                "literature_summary": lit_summary,  # Human-readable summary
            }
        else:
            # Fallback: create entry from known values
            iteration_entry = (
                f"### Iteration {current_iteration} [PARSE_FAILED]\n"
                f"**Formula**: {formula}\n"
                f"**Method**: {method}\n"
                f"**Protocol**: {safe_truncate(synthesis_protocol, 200)}\n"
                f"**PCE**: {predicted_pce}\n"
                f"**Raw Response**: {safe_truncate(response_text, 500)}"
            )
            # Parse data_context for literature references (fallback)
            literature_refs = self._extract_literature_references(data_context)
            lit_summary = self._format_literature_summary(literature_refs)
            
            structured_memory = {
                "iteration": current_iteration,
                "formula": formula,
                "method": method,
                "synthesis_protocol": synthesis_protocol,
                "pce": predicted_pce,
                "literature_refs": literature_refs,
                "literature_summary": lit_summary,
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