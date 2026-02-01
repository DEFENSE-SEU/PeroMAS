"""
MetaAgent - Chief Scientist for PSC_Agents.

Responsible for orchestrating the autonomous discovery loop for Perovskite Solar Cells.
It acts as the pure reasoning core, aligning the research workflow with scientific goals.

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
SYSTEM_PROMPT = """You are MetaAgent - Chief Scientist of PSC_Agents.

## Role
Autonomous scientific leader with full authority to guide the research process.

## Capabilities
1. **Goal Understanding**: Parse user goals, identify constraints and success criteria
2. **Hypothesis Generation**: Formulate testable scientific hypotheses each iteration
3. **Strategy Adjustment**: Dynamically adjust research direction based on results
4. **Agent Orchestration**: Assign specific tasks to downstream agents
5. **Critical Analysis**: Evaluate results, identify failures, propose corrections
6. **Memory Integration**: Learn from ALL previous iterations' complete records (formula, protocol, precursors, results)

## Downstream Agents
- DataAgent: Literature search and data extraction
- DesignAgent: Material and process design  
- FabAgent: Performance prediction
- AnalysisAgent: Gap analysis and diagnosis

## Autonomous Decision Making
- You can SKIP any agent if not needed for current iteration
- You can request agents to REDO tasks with adjusted parameters
- You can formulate NEW hypotheses based on unexpected results
- You can TERMINATE early if goal is clearly unachievable
- You can PIVOT strategy if current approach is ineffective
- You MUST use insights from memory to avoid repeating mistakes
"""


class MetaAgent(BaseAgent):
    """
    Chief Scientist agent specialized in Perovskite research planning.

    This agent functions as the 'Brain' of the system. It does not bind specific
    MCP tools but relies on pure LLM reasoning to orchestrate the workflow.

    State Updates:
        - plan: The experimental plan for the current iteration
        - is_finished: Whether the research goal has been achieved
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the MetaAgent.

        Args:
            settings: Global configuration settings.
        """
        # Direct parent initialization, no additional MCP server config needed
        super().__init__(name="MetaAgent", settings=settings)

    def _get_system_prompt(
        self,
        state: dict[str, Any],
        default_prompt: str | None = None,
    ) -> str:
        """Return the domain-specific system prompt."""
        return SYSTEM_PROMPT

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the scientific planning logic.
        """
        print(f"\n{'='*60}")
        print(f"🧠 [MetaAgent] Starting planning...")
        print(f"{'='*60}")
        
        # Show available tools (MetaAgent is pure reasoning, no tools)
        print(f"\n🛠️  Available Tools Summary:")
        print(f"   📍 Local Tools (0): None (Pure reasoning agent)")
        print(f"   🌐 MCP Tools (0): None")
        print(f"   📊 Total: 0 tools")
        
        # 1. Extract context information
        goal = state.get("goal", "")
        memory_log = state.get("memory_log", [])  # Markdown format entries
        structured_memory = state.get("structured_memory", [])  # Structured data
        current_iteration = state.get("current_iteration", 0)
        previous_analysis = state.get("analysis_report", "")
        
        print(f"\n📋 Goal: {goal}")
        print(f"🔄 Iteration: {current_iteration}")
        print(f"📚 Memory entries: {len(memory_log)}")
        print(f"📦 Structured memory: {len(structured_memory)} records")
        if previous_analysis:
            print(f"📊 Previous analysis available: Yes")

        # 2. Format memory with detailed context
        memory_summary = self._format_memory_log(memory_log)
        memory_insights = self._extract_memory_insights(structured_memory)

        # 3. Build planning prompt with strict output format
        prompt = f"""
# 🎯 RESEARCH GOAL
{goal}

# 📊 CURRENT STATUS
- **Iteration**: {current_iteration}
- **Memory Records**: {len(memory_log)} iterations archived

# 📚 COMPLETE MEMORY LOG (Your Lab Notebook)
{memory_summary}

# 🧠 KEY INSIGHTS FROM MEMORY
{memory_insights}

# 📋 PREVIOUS ITERATION ANALYSIS
{previous_analysis if previous_analysis else "None (first iteration)"}

# Your Autonomous Tasks
1. **Analyze Memory**: What do ALL previous iterations tell us? What patterns emerge?
2. **Goal Alignment Check**: Is our current approach moving toward the goal "{goal[:100]}..."?
3. **Hypothesize**: What SPECIFIC scientific hypothesis should guide this iteration?
4. **Strategy**: Should we CONTINUE current approach, PIVOT to new direction, or REFINE parameters?
5. **Plan**: Create SPECIFIC tasks for each agent (must be aligned with goal!)
6. **Decide**: Continue research or conclude?

# ⚠️ CRITICAL REQUIREMENTS:
- Your plan MUST be aligned with the original goal: "{goal[:150]}..."
- If previous iterations used wrong formulas (not matching goal), CORRECT it!
- Learn from ALL archived failures - do NOT repeat the same mistakes
- Use the complete synthesis protocol and precursor information from memory

# ⚠️ STRICT OUTPUT FORMAT (Follow Exactly)

## Step 1: Write your thinking OUTSIDE the JSON block
Thought: <your reasoning process here - analyze memory, check goal alignment, form hypothesis>

## Step 2: Output ONE JSON block with ALL required fields
```json
{{
  "memory_analysis": "<what did previous iterations teach us?>",
  "goal_alignment": "<are we on track? what needs correction?>",
  "hypothesis": "<scientific hypothesis for this iteration, aligned with goal>",
  "strategy": "<CONTINUE/PIVOT/REFINE - with specific reasoning>",
  "constraints": ["<constraint1 from goal>", "<constraint2 from memory>"],
  "agent_tasks": {{
    "DataAgent": "<specific literature search task OR 'SKIP'>",
    "DesignAgent": "<SPECIFIC material design task - include exact composition direction>",
    "FabAgent": "<specific prediction task>",
    "AnalysisAgent": "<specific analysis focus>"
  }},
  "success_criteria": "<how to judge if goal is met>"
}}
```

## Step 3: End with status on a NEW LINE (outside JSON)
FINAL_STATUS: [CONTINUE] or FINAL_STATUS: [FINISHED]

⚠️ CRITICAL RULES:
- The JSON block must be COMPLETE and VALID
- Do NOT put FINAL_STATUS inside the JSON
- Do NOT split the JSON across multiple code blocks
- DesignAgent task MUST specify composition direction based on goal analysis!
"""

        # 4. Execute pure reasoning (BaseAgent handles LLM calls)
        result = await self.autonomous_thinking(
            prompt=prompt,
            state=None,  # State already included in prompt, no need to pass again
            system_message=SYSTEM_PROMPT,
            max_iterations=1,  # Pure planning typically needs only one LLM call
        )

        # 5. Parse results
        response_text = result.get("response", "")
        is_finished = self._check_if_finished(response_text)
        
        # 6. Parse JSON plan for structured downstream access
        plan_data = self._parse_plan_json(response_text)
        
        # 7. If finished, generate final conclusion
        final_conclusion = None
        if is_finished:
            final_conclusion = await self._generate_final_conclusion(
                goal=goal,
                memory_log=memory_log,
                structured_memory=structured_memory,
                current_iteration=current_iteration,
            )

        self.logger.info(
            f"MetaAgent Planning Complete. Iteration: {current_iteration}, Finished: {is_finished}"
        )
        
        # Print output
        print(f"\n📝 [MetaAgent] Plan Output:")
        print(f"{'-'*40}")
        print(response_text)
        print(f"{'-'*40}")
        print(f"✅ Finished: {is_finished}")

        # 8. Build history entry (full, untruncated)
        history_entry = {
            "iteration": current_iteration,
            "response": response_text,  # Full response, no truncation
            "plan": plan_data if isinstance(plan_data, dict) else {"raw": str(plan_data)},
            "is_finished": is_finished,
        }

        output = {
            "plan": plan_data,  # Structured object for downstream agents
            "is_finished": is_finished,
            "meta_agent_history": [history_entry],  # Append to history
        }
        
        # Add final conclusion if finished
        if final_conclusion:
            output["final_conclusion"] = final_conclusion
            print(f"\n🏁 [MetaAgent] Final Conclusion:")
            print(f"{'-'*40}")
            print(final_conclusion)  # Full output, no truncation
            print(f"{'-'*40}")
        
        return output
    
    def _parse_plan_json(self, response_text: str) -> dict | str:
        """
        Parse JSON from response for structured downstream access.
        Returns dict if successful, otherwise returns raw text.
        """
        try:
            # Try to extract JSON from markdown code block
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            if match:
                return json.loads(match.group(1))
            
            # Try to find raw JSON object
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(response_text[start:end + 1])
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse plan JSON: {e}")
        
        # Fallback to raw text
        return response_text

    def _format_memory_log(self, memory_log: list[str]) -> str:
        """Format memory log entries to look like a comprehensive lab notebook."""
        if not memory_log:
            return "📭 No previous experiments recorded. This is the first iteration."

        formatted = ["📔 **Complete Lab Notebook:**"]
        for i, entry in enumerate(memory_log, 1):
            formatted.append(f"\n{'='*50}")
            formatted.append(f"📋 Experiment #{i}")
            formatted.append(f"{'='*50}")
            formatted.append(entry)
        formatted.append(f"\n{'='*50}")
        return "\n".join(formatted)

    def _extract_memory_insights(self, structured_memory: list[dict]) -> str:
        """
        Extract key insights from structured memory for quick reference.
        """
        if not structured_memory:
            return "No structured insights available yet."
        
        insights = ["📊 **Quick Reference:**"]
        
        # Collect all tested formulas and their results
        tested_formulas = []
        goal_aligned_count = 0
        best_pce = 0
        best_formula = "None"
        
        for record in structured_memory:
            formula = record.get("formula", "Unknown")
            pce = record.get("pce", "N/A")
            verdict = record.get("verdict", "UNKNOWN")
            aligned = record.get("aligned_with_goal", False)
            advice = record.get("advice", "")
            
            tested_formulas.append(f"  - {formula}: PCE={pce} [{verdict}]")
            
            if aligned:
                goal_aligned_count += 1
            
            # Track best PCE
            try:
                pce_value = float(str(pce).replace("%", ""))
                if pce_value > best_pce:
                    best_pce = pce_value
                    best_formula = formula
            except (ValueError, TypeError):
                pass
        
        insights.append(f"\n**Tested Formulas ({len(tested_formulas)} total):**")
        insights.extend(tested_formulas[-5:])  # Show last 5
        if len(tested_formulas) > 5:
            insights.append(f"  ... and {len(tested_formulas) - 5} more")
        
        insights.append(f"\n**Goal Alignment Rate**: {goal_aligned_count}/{len(structured_memory)} iterations")
        insights.append(f"**Best Result So Far**: {best_formula} with PCE={best_pce}%")
        
        # Latest advice from memory
        if structured_memory:
            latest = structured_memory[-1]
            if latest.get("advice"):
                insights.append(f"\n**Latest Advice from MemoryAgent**:")
                insights.append(f"  {latest.get('advice')}")
            if latest.get("learning"):
                insights.append(f"\n**Critical Learning**:")
                insights.append(f"  {latest.get('learning')}")
        
        return "\n".join(insights)

    def _check_if_finished(self, response: str) -> bool:
        """
        Robust check for finish signal using Regex.
        Matches: FINAL_STATUS: [FINISHED]
        """
        if not response:
            return False
        pattern = r"FINAL_STATUS:\s*\[FINISHED\]"
        match = re.search(pattern, response, re.IGNORECASE)
        return bool(match)

    async def _generate_final_conclusion(
        self,
        goal: str,
        memory_log: list[str],
        structured_memory: list[dict],
        current_iteration: int,
    ) -> str:
        """
        Generate a comprehensive final conclusion when the workflow finishes.
        This is the final output visible to the user.
        """
        print(f"\n{'='*60}")
        print(f"🏁 [MetaAgent] Generating Final Conclusion...")
        print(f"{'='*60}")
        
        # Extract best result from memory
        best_formula = "Unknown"
        best_pce = "N/A"
        best_protocol = ""
        best_precursors = ""
        
        for record in structured_memory:
            try:
                pce_str = str(record.get("pce", "0")).replace("%", "")
                pce_value = float(pce_str)
                current_best = float(str(best_pce).replace("%", "").replace("N/A", "0"))
                if pce_value > current_best:
                    best_pce = record.get("pce", "N/A")
                    best_formula = record.get("formula", "Unknown")
                    best_protocol = record.get("synthesis_protocol", "")
                    best_precursors = record.get("precursors", "")
            except (ValueError, TypeError):
                pass

        # Format memory for conclusion
        memory_text = self._format_memory_log(memory_log)
        
        conclusion_prompt = f"""# 🎯 RESEARCH GOAL
{goal}

# 📊 RESEARCH JOURNEY
- **Total Iterations**: {current_iteration}
- **Memory Records**: {len(memory_log)} experiments archived

# 📚 COMPLETE EXPERIMENT LOG
{memory_text}

# 🏆 BEST RESULT SUMMARY
- **Formula**: {best_formula}
- **Predicted PCE**: {best_pce}
- **Synthesis Protocol**: {best_protocol[:500] if best_protocol else 'N/A'}
- **Precursors**: {best_precursors}

# YOUR TASK: Write a Comprehensive Final Research Conclusion

As the Chief Scientist, you must now write the FINAL RESEARCH CONCLUSION that will be presented to the user.

This conclusion should include:

1. **Goal Summary**: Restate the original research goal
2. **Research Journey**: Brief summary of what was explored across all iterations
3. **Best Solution Found**:
   - Recommended formula/composition
   - Complete synthesis protocol (step-by-step)
   - Required precursors and materials
   - Expected performance metrics (PCE, Voc, etc.)
4. **Key Scientific Insights**: What did we learn about the perovskite system?
5. **Recommendations**: 
   - What should the user do next?
   - Any caveats or considerations?
6. **Confidence Assessment**: How confident are we in this recommendation?

⚠️ IMPORTANT:
- Write in clear, professional scientific language
- Include SPECIFIC details (formulas, temperatures, times, concentrations)
- This is the ONLY output the user will see for the final result
- Make it comprehensive enough to be actionable

# OUTPUT FORMAT
Write a well-structured research conclusion in Markdown format (no JSON required).
"""

        result = await self.autonomous_thinking(
            prompt=conclusion_prompt,
            state=None,
            system_message=SYSTEM_PROMPT,
            max_iterations=1,
        )
        
        return result.get("response", "Failed to generate final conclusion.")
    

if __name__ == "__main__":
    # Simple test run
    import asyncio

    async def test_meta_agent():
        test_state = {
            "goal": "How to maximize PCE of perovskite solar cells?",
            "memory_log": [],
            "structured_memory": [],
            "current_iteration": 2,
            "analysis_report": ""
        }
        
        # Must use async with to properly initialize LLM and MCP connections
        async with MetaAgent() as agent:
            result = await agent.run(test_state)
            print("MetaAgent Result:")
            print(result)

    asyncio.run(test_meta_agent())