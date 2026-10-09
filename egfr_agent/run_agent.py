import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from egfr_agent.agent import EGFRAgent


DEFAULT_QUERY = (
    "Find promising EGFR ATP competitive inhibitors. "
    "Need activity prediction, docking pose "
    "and structural analysis."
)


def save_failure_audit(request, error, stage):
    now = datetime.now(timezone.utc)
    run_id = (
        now.strftime("%Y%m%dT%H%M%S_%fZ")
        + "_"
        + uuid4().hex[:8]
    )

    directory = Path("reports/failures")
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"EGFR_Agent_Failure_{run_id}.json"

    record = {
        "run_id": run_id,
        "generated_utc": now.isoformat(),
        "status": "failed",
        "exit_code": 1,
        "failure_stage": stage,
        "user_request": request,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "trusted_decision_generated": False,
        "note": (
            "This is a failed run. Existing success "
            "reports must not be treated as its output."
        ),
    }

    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            record,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m egfr_agent.run_agent",
        description="EGFR evidence-grounded AI Agent",
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--query",
        type=str,
        help="Natural-language EGFR analysis request",
    )

    group.add_argument(
        "--query-file",
        type=Path,
        help="Read an EGFR request from a UTF-8 file",
    )

    args = parser.parse_args(argv)

    if args.query_file is not None:
        try:
            request = args.query_file.read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeError) as exc:
            parser.error(f"Cannot read query file: {exc}")
    elif args.query is not None:
        request = args.query
    else:
        request = DEFAULT_QUERY

    request = request.strip()

    if not request:
        parser.error("The analysis request cannot be empty.")

    stage = "agent_initialization"

    try:
        agent = EGFRAgent()
        stage = "agent_execution"
        agent.run(request)

    except Exception as exc:
        print(
            f"\nAGENT FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Any existing success report belongs to "
            "a previous run, not this failed run.",
            file=sys.stderr,
        )

        try:
            path = save_failure_audit(
                request, exc, stage
            )
            print(
                f"FAILURE AUDIT SAVED: {path}",
                file=sys.stderr,
            )
        except OSError as audit_error:
            print(
                "FAILURE AUDIT COULD NOT BE SAVED: "
                f"{audit_error}",
                file=sys.stderr,
            )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
