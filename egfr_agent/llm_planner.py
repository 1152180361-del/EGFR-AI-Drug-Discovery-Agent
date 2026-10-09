import json
import re

from .llm_client import OllamaClient


class LLMPlanner:

    TOOL_DESCRIPTIONS = {
        "qsar_prediction": (
            "Read an existing global ZINC QSAR prediction table. "
            "This tool does not run a new prediction and does not "
            "predict a specific EGFR_CAND identifier."
        ),
        "structure_analysis": (
            "Read previously calculated EGFR candidate evidence, "
            "including docking-related metrics and QSAR annotations. "
            "This tool does not execute a new docking calculation."
        ),
    }

    def __init__(self, memory=None):
        self.memory = memory
        self.llm = OllamaClient()

    @staticmethod
    def parse_json(raw):

        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError(
                "Ollama returned an empty planning response."
            )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()

        for match in re.finditer(r"\{", raw):
            try:
                result, _ = decoder.raw_decode(
                    raw[match.start():]
                )
            except json.JSONDecodeError:
                continue

            if isinstance(result, dict):
                return result

        raise RuntimeError(
            "LLM planning output is not valid JSON. "
            "No fallback plan was executed. "
            f"Response preview: {raw[:500]}"
        )

    @staticmethod
    def required_tools(request):

        text = request.lower()

        activity_terms = [
            "activity", "qsar", "potency",
            "predict", "prediction", "ic50",
            "活性", "预测", "效力",
        ]

        structure_terms = [
            "structure", "docking", "pose",
            "hinge", "atp", "pocket",
            "promising", "screen", "rank",
            "recommend", "compare",
            "结构", "对接", "口袋",
            "筛选", "排名", "推荐", "比较",
        ]

        required = []

        if any(term in text for term in activity_terms):
            required.append("qsar_prediction")

        if any(term in text for term in structure_terms):
            required.append("structure_analysis")

        return required

    def plan(self, user_request):

        if not isinstance(user_request, str):
            raise TypeError(
                "user_request must be a string."
            )

        if not user_request.strip():
            raise ValueError(
                "user_request cannot be empty."
            )

        # Previous recommendations are intentionally NOT
        # injected into an open-ended discovery request.
        # Otherwise, the model may anchor on a prior candidate.

        tool_catalog = json.dumps(
            self.TOOL_DESCRIPTIONS,
            indent=2,
            ensure_ascii=False,
        )

        request_json = json.dumps(
            user_request,
            ensure_ascii=False,
        )

        prompt = f"""
You are a tool-planning assistant for an EGFR
computational drug discovery workflow.

You select existing tools.
You do not invent scientific results.

AVAILABLE TOOLS:
{tool_catalog}

IMPORTANT RESTRICTIONS:

1. The QSAR tool reads a precomputed global ZINC
   prediction table. It cannot perform candidate-specific
   inference for EGFR_CAND_003 or other EGFR_CAND IDs.

2. The structure tool reads existing docking and
   structural evidence. It does not perform new docking.

3. Do not claim that either tool experimentally
   validates EGFR inhibition.

4. Do not focus on a previously recommended candidate
   unless the user explicitly requests that candidate.

5. Use only exact available tool names.

6. Select each tool no more than once.

7. Return only a JSON object with this structure:

{{
  "goal": "Brief description",
  "steps": [
    {{
      "tool": "qsar_prediction",
      "reason": "Read existing QSAR screening results"
    }}
  ]
}}

USER REQUEST:
{request_json}

Return JSON only.
"""

        raw = self.llm.generate(prompt)
        parsed = self.parse_json(raw)

        if not isinstance(parsed, dict):
            raise RuntimeError(
                "LLM plan must be a JSON object."
            )

        steps = parsed.get("steps")

        if not isinstance(steps, list):
            raise RuntimeError(
                "LLM plan must contain a steps list."
            )

        if len(steps) > 10:
            raise RuntimeError(
                "LLM returned too many planning steps."
            )

        validated = []
        seen = set()
        notes = []

        for index, step in enumerate(steps, 1):

            if not isinstance(step, dict):
                raise RuntimeError(
                    f"Invalid step {index}: expected object."
                )

            tool = step.get("tool")
            reason = step.get("reason", "")

            if tool not in self.TOOL_DESCRIPTIONS:
                raise RuntimeError(
                    f"Unknown LLM tool: {tool!r}. "
                    "Execution stopped for safety."
                )

            if tool in seen:
                notes.append(
                    f"Removed duplicate tool: {tool}"
                )
                continue

            seen.add(tool)

            if not isinstance(reason, str):
                reason = ""

            reason = reason.strip()

            # A global snapshot tool must not be described
            # as performing a candidate-specific calculation.
            candidate_mentions = re.findall(
                r"EGFR_CAND_\d+",
                reason,
                flags=re.IGNORECASE,
            )

            if candidate_mentions:
                reason = self.TOOL_DESCRIPTIONS[tool]
                notes.append(
                    "Replaced unsupported candidate-specific "
                    f"tool claim for {tool}"
                )

            if not reason:
                reason = self.TOOL_DESCRIPTIONS[tool]

            validated.append({
                "tool": tool,
                "reason": reason,
                "scope": self.TOOL_DESCRIPTIONS[tool],
            })

        for tool in self.required_tools(user_request):
            if tool not in seen:
                validated.append({
                    "tool": tool,
                    "reason": (
                        "Required by the explicit user request"
                    ),
                    "scope": self.TOOL_DESCRIPTIONS[tool],
                })
                seen.add(tool)
                notes.append(
                    f"Added required tool: {tool}"
                )

        if not validated:
            raise RuntimeError(
                "No executable tools in validated LLM plan."
            )

        goal = parsed.get("goal", "")

        if not isinstance(goal, str) or not goal.strip():
            goal = user_request.strip()

        # Avoid hallucinated candidate-specific objectives.
        if (
            re.search(
                r"EGFR_CAND_\d+",
                goal,
                flags=re.IGNORECASE,
            )
            and not re.search(
                r"EGFR_CAND_\d+",
                user_request,
                flags=re.IGNORECASE,
            )
        ):
            goal = user_request.strip()
            notes.append(
                "Removed unrequested candidate from plan goal"
            )

        return {
            "goal": goal,
            "steps": validated,
            "source": "ollama_llm_validated",
            "validation_notes": notes,
        }
