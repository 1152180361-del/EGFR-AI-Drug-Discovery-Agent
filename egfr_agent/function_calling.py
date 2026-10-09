import json

import pandas as pd

from .llm_client import OllamaClient
from .tool_scope import EGFRToolScopeResolver


class EGFRFunctionCaller:

    TOOL_SCHEMAS = [
        {
            "type": "function",
            "function": {
                "name": "qsar_prediction",
                "description": (
                    "Read an existing global ZINC QSAR "
                    "prediction snapshot. No new model "
                    "inference is performed. This tool "
                    "does not predict individual "
                    "EGFR_CAND identifiers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "structure_analysis",
                "description": (
                    "Read existing EGFR candidate "
                    "evidence, including docking pose "
                    "metrics and QSAR annotations. "
                    "Does not execute new docking."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
    ]

    SOURCE_LABELS = {
        "qsar_prediction":
            "precomputed_global_zinc_qsar_snapshot",
        "structure_analysis":
            "precomputed_egfr_candidate_evidence",
    }

    MAX_ROUNDS = 6
    MAX_REPROMPTS = 2

    def __init__(self, registry):

        self.registry = registry
        self.client = OllamaClient()
        self.scope_resolver = (
            EGFRToolScopeResolver()
        )

        self.schema_by_name = {
            item["function"]["name"]: item
            for item in self.TOOL_SCHEMAS
        }

        for name in self.schema_by_name:
            if name not in registry:
                raise RuntimeError(
                    f"Registered tool missing: {name}"
                )

    @staticmethod
    def _parse_arguments(value):

        if value is None or value == "":
            return {}

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid tool arguments JSON."
                ) from exc

        if not isinstance(value, dict):
            raise RuntimeError(
                "Tool arguments must be an object."
            )

        if value:
            raise RuntimeError(
                "Unexpected tool arguments: "
                + str(value)
            )

        return {}

    @classmethod
    def _summarize_frame(cls, name, frame):

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"{name} did not return a DataFrame."
            )

        if frame.empty:
            raise RuntimeError(
                f"{name} returned empty evidence."
            )

        records = json.loads(
            frame.head(5).to_json(
                orient="records"
            )
        )

        return {
            "source": cls.SOURCE_LABELS[name],
            "rows_returned": int(len(frame)),
            "columns": list(frame.columns),
            "sample_rows": records,
            "scientific_limitations": (
                "Precomputed computational evidence; "
                "not a new prediction, new docking run, "
                "or experimental validation."
            ),
        }

    def run(self, user_request):

        scope = self.scope_resolver.resolve(
            user_request
        )

        allowed_names = scope["allowed_tools"]
        allowed_set = set(allowed_names)

        system_prompt = f"""
You control read-only EGFR computational tools.

Only the following tools are authorized
for THIS request:

{json.dumps(allowed_names)}

Do not attempt to call any other tool.

The user's explicit exclusions must be respected.

The QSAR tool reads global precomputed ZINC
predictions. It cannot freshly predict a
specific EGFR_CAND compound.

The structure tool reads existing candidate
evidence. It does not run new docking.

Do not claim experimental inhibition,
verified potency or validated hydrogen bonds.

Use native function calling to fulfill
the authorized task.

Each tool accepts an empty argument object.

Call every authorized tool that is necessary
for the request, at most once.

Do not treat global ZINC predictions and
EGFR_CAND rows as the same compounds
without an explicit identifier match.

Once tool results arrive, summarize the
relevant evidence without inventing results.

The final candidate ranking is determined
by a separate Python decision engine.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_request,
            },
        ]

        executed = {}
        requested_by_model = []
        reprompts = 0

        call_log = [
            {
                "event": "scope_resolved",
                "authorization_method": (
                    scope["authorization_method"]
                ),
                "allowed_tools": allowed_names,
                "prohibited_tools": (
                    scope["prohibited_tools"]
                ),
            }
        ]

        print("\nTOOL SCOPE GUARD")
        print("Allowed:", allowed_names)
        print(
            "Unavailable for this request:",
            scope["prohibited_tools"],
        )

        for round_number in range(
            1,
            self.MAX_ROUNDS + 1,
        ):

            remaining = [
                name
                for name in allowed_names
                if name not in executed
            ]

            available_schemas = [
                self.schema_by_name[name]
                for name in remaining
            ]

            response = self.client.chat(
                messages,
                tools=(
                    available_schemas
                    if available_schemas
                    else None
                ),
            )

            message = response.get("message")

            if not isinstance(message, dict):
                raise RuntimeError(
                    "Invalid Ollama assistant message."
                )

            messages.append(message)

            calls = message.get("tool_calls") or []

            if not isinstance(calls, list):
                raise RuntimeError(
                    "Invalid tool_calls format."
                )

            if not calls:

                missing = [
                    name
                    for name in allowed_names
                    if name not in executed
                ]

                if missing:

                    if reprompts >= self.MAX_REPROMPTS:
                        raise RuntimeError(
                            "LLM did not invoke the "
                            "authorized required tools: "
                            + ", ".join(missing)
                        )

                    reprompts += 1

                    call_log.append({
                        "round": round_number,
                        "event": "coverage_reprompt",
                        "missing_tools": missing,
                    })

                    messages.append({
                        "role": "user",
                        "content": (
                            "The task is not complete. "
                            "Use native function calls "
                            "for these authorized tools: "
                            + ", ".join(missing)
                            + ". Do not only describe "
                            "the calls."
                        ),
                    })

                    continue

                return {
                    "model": self.client.model,
                    "mode": (
                        "ollama_native_tool_calling"
                    ),
                    "rounds": round_number,
                    "required_tools": allowed_names,
                    "executed_tools": list(
                        executed.keys()
                    ),
                    "model_requested_tools": (
                        requested_by_model
                    ),
                    "scope": scope,
                    "results": executed,
                    "call_log": call_log,
                    "llm_summary": (
                        message.get("content")
                        or ""
                    ),
                }

            # First validate every request in the batch.
            # Invalid calls must not be executed.
            validated_calls = []

            for call in calls:

                if not isinstance(call, dict):
                    raise RuntimeError(
                        "Invalid tool call object."
                    )

                function = call.get("function")

                if not isinstance(function, dict):
                    raise RuntimeError(
                        "Tool call missing function."
                    )

                name = function.get("name")

                if name not in allowed_set:
                    raise RuntimeError(
                        "Rejected unapproved LLM tool: "
                        + str(name)
                    )

                self._parse_arguments(
                    function.get("arguments", {})
                )

                validated_calls.append(name)

            for name in validated_calls:

                requested_by_model.append(name)

                if name in executed:

                    call_log.append({
                        "round": round_number,
                        "tool": name,
                        "status": "duplicate_blocked",
                    })

                    messages.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps({
                            "status": "not_reexecuted",
                            "reason": (
                                "Tool already executed."
                            ),
                        }),
                    })

                    continue

                print(
                    "\nNATIVE FUNCTION CALL:",
                    name,
                )

                result = self.registry[name].run()

                summary = self._summarize_frame(
                    name,
                    result,
                )

                executed[name] = result

                call_log.append({
                    "round": round_number,
                    "tool": name,
                    "status": "executed",
                    "source": summary["source"],
                    "rows_returned": (
                        summary["rows_returned"]
                    ),
                })

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(
                        summary,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                })

                print(
                    "SOURCE:",
                    summary["source"],
                )

                print(
                    "ROWS:",
                    summary["rows_returned"],
                )

                print(
                    result.head(3).to_string(
                        index=False
                    )
                )

        raise RuntimeError(
            "Function-calling loop exceeded "
            f"{self.MAX_ROUNDS} rounds."
        )
