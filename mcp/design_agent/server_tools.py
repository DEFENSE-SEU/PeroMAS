"""
Server Tools Interface for DesignAgent

Manages tools that run on remote servers (MatterGen, CSLLM).
Supports two modes:
1. Interactive Mode: Wait for user to input server results via terminal
2. Mock Mode: Use LLM to generate scientifically plausible results (model from LLM_MODEL_ID)

Mock 模式现在由 LLM 驱动，可以全自动运行 workflow。

Author: PSC_Agents Team
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

# For LLM-assisted mock mode
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class ToolMode(Enum):
    """Tool execution mode"""
    INTERACTIVE = "interactive"  # Wait for user input from server
    MOCK = "mock"               # Use LLM to generate results (fully automatic)


# Global LLM client for mock mode (lazy initialization)
_mock_llm_client: Optional["OpenAI"] = None

# Mock 模式使用通用 LLM 配置（与所有 Agent 一致）
MOCK_LLM_MODEL = os.getenv("LLM_MODEL_ID", "openai/gpt-5.4")


def _get_mock_llm_client() -> Optional["OpenAI"]:
    """Get LLM client for mock mode using the global LLM config from .env."""
    global _mock_llm_client
    if _mock_llm_client is None and HAS_OPENAI:
        api_key = os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "")

        if api_key and base_url:
            _mock_llm_client = OpenAI(api_key=api_key, base_url=base_url)
    return _mock_llm_client


@dataclass
class ServerToolConfig:
    """Configuration for server tools"""
    mode: ToolMode = ToolMode.MOCK  # Default to mock (LLM-driven)
    timeout_seconds: int = 600
    
    @classmethod
    def from_mode_str(cls, mode_str: str) -> "ServerToolConfig":
        """Create config from mode string"""
        mode = ToolMode.INTERACTIVE if mode_str.lower() == "interactive" else ToolMode.MOCK
        return cls(mode=mode)


class ServerTool(ABC):
    """Base class for server-side tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.config = ServerToolConfig()
    
    def set_mode(self, mode: ToolMode):
        """Set execution mode"""
        self.config.mode = mode
    
    @abstractmethod
    def get_command_hint(self, args: dict[str, Any]) -> str:
        """Get command hint for running on server"""
        pass
    
    @abstractmethod
    def get_llm_prompt(self, args: dict[str, Any]) -> str:
        """Get prompt for LLM to generate scientifically plausible result"""
        pass
    
    @abstractmethod
    def parse_server_result(self, raw_result: str) -> dict[str, Any]:
        """Parse result from server output"""
        pass
    
    def execute(self, args: dict[str, Any]) -> str:
        """Execute tool based on current mode"""
        if self.config.mode == ToolMode.MOCK:
            return self._execute_mock(args)
        else:
            return self._execute_interactive(args)
    
    def _execute_mock(self, args: dict[str, Any]) -> str:
        """Execute in mock mode - use LLM to generate scientifically plausible results"""
        client = _get_mock_llm_client()
        if client is None:
            return json.dumps({
                "status": "error",
                "error": "LLM client not available. Check GOOGLE_API_KEY and GOOGLE_BASE_URL.",
                "_mode": "mock"
            }, indent=2, ensure_ascii=False)
        
        prompt = self.get_llm_prompt(args)
        
        # 最多重试3次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"   🧠 LLM generating {self.name} result...")
                response = client.chat.completions.create(
                    model=MOCK_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a perovskite materials science expert. Generate realistic predictions. IMPORTANT: Respond with ONLY a compact JSON object on a single line if possible. No markdown, no explanations, just pure JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4096,  # 增加 token 限制
                )
                
                raw_result = response.choices[0].message.content.strip()
                
                # Check if response was truncated (no closing brace)
                if raw_result.count('{') > raw_result.count('}'):
                    print(f"   ⚠️  Response truncated (attempt {attempt + 1}/{max_retries}), retrying...")
                    continue
                
                # Remove markdown code blocks if present
                if raw_result.startswith("```"):
                    import re
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_result)
                    if match:
                        raw_result = match.group(1)
                
                # Parse JSON
                try:
                    result = json.loads(raw_result)
                    result["_mode"] = "mock_llm"
                    result["_model"] = MOCK_LLM_MODEL
                    return json.dumps(result, indent=2, ensure_ascii=False)
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        continue
                    # 最后一次尝试失败，返回原始内容
                    return json.dumps({
                        "status": "success",
                        "tool": self.name,
                        "result": raw_result,  # 完整返回
                        "_mode": "mock_llm",
                        "_model": MOCK_LLM_MODEL,
                        "_note": f"LLM response was not valid JSON: {str(e)}"
                    }, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"   ⚠️  LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
        
        # 所有重试都失败
        return json.dumps({
            "status": "error",
            "error": "All retries failed",
            "_mode": "mock"
        }, indent=2, ensure_ascii=False)
    
    def _execute_interactive(self, args: dict[str, Any]) -> str:
        """Execute in interactive mode - wait for user input"""
        command_hint = self.get_command_hint(args)
        
        print()
        print("╔" + "═" * 70 + "╗")
        print(f"║{'🖥️  SERVER TOOL: ' + self.name:^70}║")
        print("╠" + "═" * 70 + "╣")
        print("║" + " Please run the following command on the server:".ljust(70) + "║")
        print("╠" + "─" * 70 + "╣")
        
        # Print command (may be multi-line)
        for line in command_hint.split('\n'):
            if len(line) > 68:
                while len(line) > 68:
                    print(f"║ {line[:68]} ║")
                    line = line[68:]
                if line:
                    print(f"║ {line:<68} ║")
            else:
                print(f"║ {line:<68} ║")
        
        print("╠" + "─" * 70 + "╣")
        print("║" + " After execution, paste the result below.".ljust(70) + "║")
        print("║" + " Enter 'END' on a new line when done, or 'SKIP' to use LLM mock.".ljust(70) + "║")
        print("╚" + "═" * 70 + "╝")
        print()
        
        # Collect multi-line input
        lines = []
        print("📥 Paste server result (END to finish, SKIP for LLM mock):")
        print("-" * 40)
        
        while True:
            try:
                line = input()
                if line.strip().upper() == "END":
                    break
                elif line.strip().upper() == "SKIP":
                    print("⏭️  Skipped. Using LLM mock...")
                    return self._execute_mock(args)
                lines.append(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n⚠️  Cancelled. Using mock data...")
                return self._execute_mock(args)
        
        raw_result = "\n".join(lines)
        
        if not raw_result.strip():
            print("⚠️  Empty input. Using mock data...")
            return self._execute_mock(args)
        
        # Parse and return result
        try:
            parsed = self.parse_server_result(raw_result)
            parsed["_mode"] = "interactive"
            parsed["_timestamp"] = datetime.now().isoformat()
            print("✅ Server result received and parsed.")
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Failed to parse result: {e}")
            print("   Returning raw result...")
            return json.dumps({
                "_mode": "interactive",
                "_raw": raw_result,
                "_parse_error": str(e)
            }, indent=2, ensure_ascii=False)


# =============================================================================
# MatterGen Tool - Material Structure Generation
# =============================================================================

class MatterGenTool(ServerTool):
    """MatterGen material structure generation tool"""
    
    def __init__(self):
        super().__init__(
            name="generate_material_structure",
            description="Generate perovskite crystal structures using MatterGen model"
        )
    
    def get_command_hint(self, args: dict[str, Any]) -> str:
        """Generate command for running MatterGen on server"""
        # Build command line arguments
        cmd_args = []
        
        if "target_pce" in args:
            cmd_args.append(f"--pce {args['target_pce']}")
        if "target_voc" in args:
            cmd_args.append(f"--voc {args['target_voc']}")
        if "target_jsc" in args:
            cmd_args.append(f"--jsc {args['target_jsc']}")
        if "target_ff" in args:
            cmd_args.append(f"--ff {args['target_ff']}")
        if "target_bandgap" in args:
            cmd_args.append(f"--dft_band_gap {args['target_bandgap']}")
        if "stability_threshold" in args:
            cmd_args.append(f"--energy_above_hull {args['stability_threshold']}")
        
        # Default to PCE 25 if no properties specified
        if not cmd_args:
            cmd_args.append("--pce 25.0")
        
        num_candidates = args.get("num_candidates", 5)
        batch_size = args.get("batch_size", 30)
        num_batches = args.get("num_batches", 1)
        guidance = args.get("diffusion_guidance_factor", 8.0)
        
        props_str = " ".join(cmd_args)
        
        command = f"""# On server with MatterGen environment:
cd /seu_share2/home/xxx/psc_mattergen
conda activate mattergen

# Generate perovskite structures with property constraints
python scripts/test_generate.py {props_str} \\
    --batch_size {batch_size} \\
    --num_batches {num_batches} \\
    --diffusion_guidance_factor {guidance}

# Output includes:
# - generated_crystals.extxyz (structure file)
# - cif_files/ (individual CIF files)
# - generate_config.json (parameters used)
#
# Copy the formulas from terminal output, e.g.:
# 📋 Designed Formulas (30 structures):
# ├── CsPbI3
# ├── FAPbI3
# └── ..."""
        return command
    
    def get_llm_prompt(self, args: dict[str, Any]) -> str:
        """Generate LLM prompt for material structure prediction"""
        target_pce = args.get("target_pce", 25.0)
        target_bandgap = args.get("target_bandgap", 1.5)
        target_voc = args.get("target_voc")
        target_jsc = args.get("target_jsc")
        target_ff = args.get("target_ff")
        stability_threshold = args.get("stability_threshold", 0.05)
        num_candidates = args.get("num_candidates", 5)
        
        # 构建约束条件
        constraints = []
        if target_pce:
            constraints.append(f"PCE≥{target_pce}%")
        if target_bandgap:
            constraints.append(f"Eg~{target_bandgap}eV")
        if target_voc:
            constraints.append(f"Voc≥{target_voc}V")
        if target_jsc:
            constraints.append(f"Jsc≥{target_jsc}mA/cm²")
        if target_ff:
            constraints.append(f"FF≥{target_ff}%")
        constraints_str = ", ".join(constraints) if constraints else f"PCE≥{target_pce}%"
        
        return f"""Design {num_candidates} perovskite solar cell compositions targeting: {constraints_str}

Knowledge base:
- A-site: Cs (stability), FA/MA (bandgap tuning), Rb (defect passivation)
- B-site: Pb (high PCE), Sn (eco-friendly, narrow Eg), mixed Pb-Sn (tandem)
- X-site: I (narrow Eg~1.5eV), Br (wide Eg~2.3eV), mixed I-Br (tunable)
- Doping: Mn/Zn/Cd (stability), Bi (defect tolerance)
- Literature benchmarks: FAPbI3~25.7%, FA0.95Cs0.05PbI3~25.2%, FASnI3~14.8%

Return ONLY valid JSON:
{{"status":"success","candidates":[{{"formula":"FA0.95Cs0.05PbI3","structure_type":"3D","predicted_pce":25.2,"predicted_bandgap":1.51,"energy_above_hull":0.012,"reasoning":"Cs stabilizes black phase"}}],"generation_params":{{"target_pce":{target_pce},"num_candidates":{num_candidates}}}}}

Generate {num_candidates} DIFFERENT realistic formulas. JSON only, no markdown."""
    
    def parse_server_result(self, raw_result: str) -> dict[str, Any]:
        """Parse MatterGen output"""
        # Try to parse as JSON first
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            pass
        
        # Try to extract formulas from text output
        candidates = []
        lines = raw_result.strip().split('\n')
        
        import re
        for line in lines:
            # Look for formula patterns (e.g., "1. Cs0.05FA0.95PbI3")
            line = line.strip()
            if not line:
                continue
            
            # Remove common prefixes/numbering
            # Patterns: "1. Formula", "├── Formula", "│   Formula", "- Formula"
            clean_line = re.sub(r'^(\d+\.\s*|[├└│]──\s*|[├└│]\s*|-\s*)', '', line).strip()
            
            if not clean_line:
                continue
            
            # Check if it looks like a perovskite formula
            # Common elements: MA, FA, Cs, Rb, Pb, Sn, Ge, Bi, Ag, I, Br, Cl
            if any(elem in clean_line for elem in ['Pb', 'Sn', 'Bi', 'Ge', 'MA', 'FA', 'Cs']):
                # Further validation: should have halide (I, Br, Cl)
                if any(halide in clean_line for halide in ['I', 'Br', 'Cl']):
                    candidates.append({
                        "formula": clean_line,
                        "structure_type": "3D",
                        "source": "MatterGen_server"
                    })
        
        # Build clear result message
        if candidates:
            formula_list = [c["formula"] for c in candidates]
            instruction = f"⚠️ IMPORTANT: Use these EXACT formulas for subsequent tool calls: {formula_list}"
            return {
                "status": "success",
                "tool": "MatterGen",
                "candidates": candidates,
                "generated_formulas": formula_list,
                "instruction": instruction,
                "note": "You MUST use these exact formulas for check_synthesizability and other tools"
            }
        else:
            return {
                "status": "warning",
                "tool": "MatterGen",
                "candidates": [],
                "raw_output": raw_result,
                "note": "Could not parse formulas from server output. Please check the raw output."
            }


# =============================================================================
# CSLLM Tools - Synthesis, Method, Precursor Prediction
# =============================================================================

class CSLLMSynthesisTool(ServerTool):
    """CSLLM synthesizability prediction tool"""
    
    def __init__(self):
        super().__init__(
            name="check_synthesizability",
            description="Check if a material can be synthesized using CSLLM"
        )
    
    def get_command_hint(self, args: dict[str, Any]) -> str:
        formula = args.get("formula", "MAPbI3")
        return f"""# On server with CSLLM environment:
cd /seu_share2/home/xxx/CSLLM
conda activate csllm

# Test synthesizability for single formula
python test_perovskite_models.py --model synthesis --formula "{formula}"

# Or compare with original model
python test_perovskite_models.py --model synthesis --formula "{formula}" --compare

# Expected output format:
# ┌─────────────────────────────────────────┐
# │ Formula: {formula}
# │ Synthesizable: True/False
# │ Confidence: 0.95
# │ Reasoning: ...
# └─────────────────────────────────────────┘"""
    
    def get_llm_prompt(self, args: dict[str, Any]) -> str:
        """Generate LLM prompt for synthesizability prediction"""
        formula = args.get("formula", "Unknown")
        structure_type = args.get("structure_type", "3D")
        
        return f"""Evaluate synthesizability of perovskite: {formula} (structure: {structure_type})

Analysis criteria:
1. Goldschmidt tolerance factor t: 0.8<t<1.0 for 3D perovskite
2. Octahedral factor μ: 0.44<μ<0.90 for stable octahedra
3. Phase stability: α-phase preferred, δ-phase is non-perovskite
4. Precursor availability: PbI2, SnI2, CsI, FAI, MAI are commercial
5. Known synthesis: Check if similar compositions reported in literature
6. Challenges: Sn²⁺ oxidation, phase segregation in mixed-halide, moisture sensitivity

Examples:
- CsPbI3: t=0.81, synthesizable but δ-phase at RT, needs additives
- FAPbI3: t=0.99, needs Cs/MA to stabilize α-phase
- CsSnI3: synthesizable but Sn²⁺ easily oxidizes to Sn⁴⁺

Return ONLY valid JSON:
{{"status":"success","formula":"{formula}","structure_type":"{structure_type}","synthesizable":true,"confidence":0.85,"tolerance_factor":0.91,"reasoning":"scientific justification","challenges":["specific challenge"],"recommended_additives":["additive if needed"]}}

JSON only, no markdown."""
    
    def parse_server_result(self, raw_result: str) -> dict[str, Any]:
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            pass
        
        # Parse text output for synthesizability
        result_lower = raw_result.lower()
        
        # Check for positive indicators
        positive_patterns = ["yes", "true", "can be synthesized", "synthesizable", "is possible", "可以合成", "可合成"]
        negative_patterns = ["no", "false", "cannot", "not synthesizable", "impossible", "不可合成", "无法合成"]
        
        synthesizable = None
        for pattern in positive_patterns:
            if pattern in result_lower:
                synthesizable = True
                break
        if synthesizable is None:
            for pattern in negative_patterns:
                if pattern in result_lower:
                    synthesizable = False
                    break
        
        # If still unclear, default to None and let LLM interpret
        return {
            "status": "success",
            "tool": "CSLLM-Synthesis",
            "synthesizable": synthesizable,
            "confidence": 0.8 if synthesizable is not None else 0.5,
            "raw_output": raw_result,
            "note": "This is the actual server result. Base your analysis on this data."
        }


class CSLLMMethodTool(ServerTool):
    """CSLLM synthesis method prediction tool"""
    
    def __init__(self):
        super().__init__(
            name="predict_synthesis_method",
            description="Predict synthesis method (solution/solid_state) using CSLLM"
        )
    
    def get_command_hint(self, args: dict[str, Any]) -> str:
        formula = args.get("formula", "MAPbI3")
        return f"""# On server with CSLLM environment:
cd /seu_share2/home/xxx/CSLLM
conda activate csllm

# Predict synthesis method for single formula
python test_perovskite_models.py --model method --formula "{formula}"

# Or compare with original model
python test_perovskite_models.py --model method --formula "{formula}" --compare

# Expected output format:
# ┌─────────────────────────────────────────┐
# │ Formula: {formula}
# │ Method: [Precursor Solution] ...
# │         [Spin Coating] ...
# │         [Annealing] ...
# │ Confidence: 0.92
# └─────────────────────────────────────────┘"""
    
    def get_llm_prompt(self, args: dict[str, Any]) -> str:
        """Generate LLM prompt for synthesis method prediction"""
        formula = args.get("formula", "MAPbI3")
        
        return f"""Predict the complete synthesis protocol for perovskite: {formula}

You must provide a detailed, step-by-step synthesis method as a SINGLE comprehensive text in "synthesis_protocol".

Knowledge base:
- Solvents: DMF, DMSO, NMP, GBL. Typical ratio DMF:DMSO = 4:1
- Anti-solvents: chlorobenzene, toluene, diethyl ether (drip at 5-10s before spin ends)
- Spin-coating: 1000-6000 rpm, 30-60s, multi-step possible
- Annealing: 100-150°C for organic-inorganic, 170-350°C for all-inorganic CsPbX3
- For CsPbI3: needs high-temp annealing (>250°C) or additive-assisted low-temp route

Return ONLY valid JSON with a COMPLETE synthesis_protocol field:
{{
  "status": "success",
  "formula": "{formula}",
  "method": "one-step spin-coating",
  "synthesis_protocol": "Step 1 (Precursor Preparation): Dissolve PbI2 (1.2 M) and CsI (1.2 M) in DMF:DMSO (4:1 v/v) mixed solvent. Add 5 mol% excess CsI to compensate volatilization. Heat at 70°C and stir for 2 hours until fully dissolved. Filter through 0.22 μm PTFE filter. Step 2 (Substrate Preparation): Clean ITO/glass substrates sequentially with detergent, DI water, acetone, and isopropanol (15 min each, ultrasonic). UV-ozone treat for 15 min before use. Step 3 (Film Deposition): Spin-coat PTAA hole transport layer at 6000 rpm for 30s. Pre-heat substrate at 70°C. Deposit 50 μL precursor solution at 1000 rpm for 10s then 4000 rpm for 30s. Drip 200 μL chlorobenzene anti-solvent at 20s during high-speed spinning. Step 4 (Annealing): Immediately transfer to hotplate at 100°C for 10 min, then ramp to 170°C for 15 min to induce δ-to-α phase transition. Cool naturally. Step 5 (Post-treatment): Optional surface passivation with PEAI (2 mg/mL in IPA) spin-coated at 5000 rpm.",
  "confidence": 0.9
}}

The synthesis_protocol MUST be a single comprehensive paragraph with all steps. JSON only, no markdown."""
    
    def parse_server_result(self, raw_result: str) -> dict[str, Any]:
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            pass
        
        result_lower = raw_result.lower()
        
        # Detect synthesis method from output
        if "solid" in result_lower or "solid_state" in result_lower or "solid-state" in result_lower:
            method = "solid_state"
        elif "solution" in result_lower or "spin" in result_lower or "precursor" in result_lower:
            method = "solution"
        else:
            method = "unknown"
        
        # Extract key steps if present
        steps = []
        for keyword in ["precursor", "spin", "coating", "annealing", "deposition", "film"]:
            if keyword in result_lower:
                steps.append(keyword)
        
        return {
            "status": "success",
            "tool": "CSLLM-Method",
            "method": method,
            "detected_steps": steps,
            "raw_output": raw_result,
            "note": "This is the actual server result. Base your analysis on this data."
        }


class CSLLMPrecursorTool(ServerTool):
    """CSLLM precursor prediction tool"""
    
    def __init__(self):
        super().__init__(
            name="predict_precursors",
            description="Predict precursor chemicals for synthesis using CSLLM"
        )
    
    def get_command_hint(self, args: dict[str, Any]) -> str:
        formula = args.get("formula", "MAPbI3")
        method = args.get("synthesis_method", "solution")
        return f"""# On server with CSLLM environment:
cd /seu_share2/home/xxx/CSLLM
conda activate csllm

# Predict precursors for single formula
python test_perovskite_models.py --model precursor --formula "{formula}"

# Or compare with original model
python test_perovskite_models.py --model precursor --formula "{formula}" --compare

# Expected output format:
# ┌─────────────────────────────────────────┐
# │ Formula: {formula}
# │ Precursors: ['PbI2', 'FAI', 'CsI', ...]
# │ Solvents: ['DMF', 'DMSO']
# │ Concentrations: {{'PbI2': '1.2M', ...}}
# │ Confidence: 0.88
# └─────────────────────────────────────────┘"""
    
    def get_llm_prompt(self, args: dict[str, Any]) -> str:
        """Generate LLM prompt for precursor prediction"""
        formula = args.get("formula", "MAPbI3")
        method = args.get("synthesis_method", "spin-coating")
        
        return f"""Predict precursors for perovskite synthesis: {formula} (method: {method})

Precursor knowledge base:
- A-site sources: MAI (CH3NH3I), FAI (HC(NH2)2I), CsI, RbI
- B-site sources: PbI2, PbBr2, PbCl2, SnI2, SnF2 (Sn stabilizer)
- Additives: MACl (grain growth), NH4Cl (orientation), MASCN (defect passivation)
- Solvents: DMF (main), DMSO (Lewis base, improves morphology), NMP, GBL
- Typical ratios: DMF:DMSO = 4:1 or 9:1, concentration 1.0-1.5M

Stoichiometry rules:
- ABX3: A:B:X = 1:1:3 (slight excess of AX improves crystallization)
- Mixed A-site (FA0.95Cs0.05PbI3): FAI:CsI:PbI2 = 0.95:0.05:1
- For Sn: add 10-20% SnF2 to prevent Sn²⁺ oxidation

Return ONLY valid JSON:
{{"status":"success","formula":"{formula}","precursors":[{{"name":"PbI2","amount":"1.2M","role":"B-site"}},{{"name":"FAI","amount":"1.26M","role":"A-site (5% excess)"}}],"solvents":["DMF","DMSO"],"solvent_ratio":"4:1","total_concentration":"1.2M","additives":[{{"name":"MACl","amount":"30mol%","purpose":"grain enlargement"}}],"preparation_notes":"Dissolve at 70°C, stir 2h, filter before use","confidence":0.88}}

JSON only, no markdown."""
    
    def parse_server_result(self, raw_result: str) -> dict[str, Any]:
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            pass
        
        # Try to extract precursor names from text
        import re
        
        # Common precursor patterns
        precursors = []
        common_precursors = [
            'PbI2', 'PbBr2', 'PbCl2', 'SnI2', 'SnF2', 'SnCl2',
            'MAI', 'MABr', 'MACl', 'FAI', 'FABr', 'FACl',
            'CsI', 'CsBr', 'CsCl', 'RbI', 'KI',
            'DMF', 'DMSO', 'NMP', 'GBL'
        ]
        
        raw_upper = raw_result.upper()
        for prec in common_precursors:
            if prec.upper() in raw_upper:
                precursors.append(prec)
        
        # Try to extract list patterns like ['PbI2', 'FAI']
        match = re.search(r'\[.*?\]', raw_result)
        if match:
            try:
                extracted = json.loads(match.group(0))
                if isinstance(extracted, list):
                    precursors = extracted
            except:
                pass
        
        return {
            "status": "success",
            "tool": "CSLLM-Precursor",
            "detected_precursors": precursors if precursors else ["Could not auto-detect"],
            "raw_output": raw_result,
            "note": "This is the actual server result. Base your analysis on this data."
        }


# =============================================================================
# Server Tool Manager
# =============================================================================

class ServerToolManager:
    """
    Manager for all server-side tools used by DesignAgent.
    
    Usage:
        manager = ServerToolManager(mode="mock")  # or "interactive"
        
        # Execute a tool
        result = manager.execute("generate_material_structure", {"target_pce": 25.0})
        
        # Switch mode
        manager.set_mode("interactive")
    """
    
    def __init__(self, mode: str = "llm"):
        """
        Initialize tool manager.
        
        Args:
            mode: "mock" or "interactive"
                  - mock: Use LLM (Gemini Flash) to generate results (fully automatic)
                  - interactive: Wait for real server input (manual)
        """
        self._mode = ToolMode.INTERACTIVE if mode.lower() == "interactive" else ToolMode.MOCK
        
        # Initialize all tools
        # - MatterGen: generate_material_structure
        # - CSLLM: check_synthesizability, predict_synthesis_method, predict_precursors
        self._tools: dict[str, ServerTool] = {
            "generate_material_structure": MatterGenTool(),
            "check_synthesizability": CSLLMSynthesisTool(),
            "predict_synthesis_method": CSLLMMethodTool(),
            "predict_precursors": CSLLMPrecursorTool(),
        }
        
        # Set initial mode for all tools
        self.set_mode(mode)
    
    @property
    def mode(self) -> str:
        return self._mode.value
    
    def set_mode(self, mode: str):
        """Set mode for all tools"""
        self._mode = ToolMode.INTERACTIVE if mode.lower() == "interactive" else ToolMode.MOCK
        for tool in self._tools.values():
            tool.set_mode(self._mode)
        print(f"🔧 Server tools mode: {self._mode.value} ({'LLM-driven' if self._mode == ToolMode.MOCK else 'Server interaction'})")
    
    def get_tool_names(self) -> list[str]:
        """Get list of available tool names"""
        return list(self._tools.keys())
    
    def has_tool(self, name: str) -> bool:
        """Check if tool exists"""
        return name in self._tools
    
    def execute(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name"""
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        
        tool = self._tools[name]
        return tool.execute(args)
    
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function schemas for all tools"""
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "generate_material_structure",
                    "description": "Generate candidate perovskite structures using MatterGen model based on target properties.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_pce": {"type": "number", "description": "Target PCE in %"},
                            "target_voc": {"type": "number", "description": "Target Voc in V"},
                            "target_jsc": {"type": "number", "description": "Target Jsc in mA/cm²"},
                            "target_ff": {"type": "number", "description": "Target FF in %"},
                            "target_bandgap": {"type": "number", "description": "Target band gap in eV"},
                            "stability_threshold": {"type": "number", "description": "Max energy above hull in eV/atom"},
                            "num_candidates": {"type": "integer", "description": "Number of candidates to generate"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_synthesizability",
                    "description": "Check if a perovskite formula can be synthesized using CSLLM model.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "formula": {"type": "string", "description": "Chemical formula, e.g., MAPbI3"},
                            "structure_type": {"type": "string", "description": "Structure type: 3D, 2D, 0D, quasi-2D"}
                        },
                        "required": ["formula"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "predict_synthesis_method",
                    "description": "Predict synthesis method (solution or solid_state) for a perovskite using CSLLM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "formula": {"type": "string", "description": "Chemical formula"}
                        },
                        "required": ["formula"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "predict_precursors",
                    "description": "Predict precursor chemicals for perovskite synthesis using CSLLM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "formula": {"type": "string", "description": "Target material formula"},
                            "synthesis_method": {"type": "string", "description": "Synthesis method: spin-coating, blade-coating, evaporation, any"}
                        },
                        "required": ["formula"]
                    }
                }
            }
            # NOTE: design_process_parameters removed - not supported by actual CSLLM server
            # Available CSLLM tools: check_synthesizability, predict_synthesis_method, predict_precursors
        ]
        return schemas


# =============================================================================
# Convenience function
# =============================================================================

def create_server_tool_manager(mode: str = "mock") -> ServerToolManager:
    """Create a server tool manager with specified mode"""
    return ServerToolManager(mode=mode)
