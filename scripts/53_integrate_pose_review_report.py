from pathlib import Path
from datetime import datetime
import os

ROOT = Path(__file__).resolve().parents[1]

TARGET = ROOT / "egfr_agent/report_generator.py"

BACKUP_DIR = ROOT / "reports/code_backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

source = TARGET.read_text(encoding="utf-8")

IMPORT_OLD = (
    "from .risk_engine import EGFRAgentRiskEngine\n"
)

IMPORT_NEW = (
    IMPORT_OLD
    + "from .pose_review_evidence import render_pose_review\n"
)

RETURN_OLD = '        return "\\n".join(lines)'

RETURN_NEW = '''        # Independent, non-scoring manual pose review evidence.
        # Report only candidates present in this ranking.
        review_ids = (
            ranking_df["candidate"]
            .head(10)
            .astype(str)
            .tolist()
        )

        review_text = render_pose_review(review_ids)

        review_text = review_text.replace(
            "## Manual pose review evidence",
            "### 6.1 Manual pose review evidence",
            1,
        )

        boundary = "## 7. Workflow classification"

        if lines.count(boundary) != 1:
            raise RuntimeError(
                "Report section boundary changed."
            )

        pos = lines.index(boundary)

        lines[pos:pos] = (
            review_text.splitlines() + [""]
        )

        return "\\n".join(lines)'''

if "from .pose_review_evidence import render_pose_review" in source:
    raise RuntimeError(
        "Pose review already integrated. "
        "Refusing duplicate modification."
    )

if source.count(IMPORT_OLD) != 1:
    raise RuntimeError("Import anchor not unique")

if source.count(RETURN_OLD) != 1:
    raise RuntimeError("Return anchor not unique")

patched = source.replace(
    IMPORT_OLD, IMPORT_NEW, 1
)

patched = patched.replace(
    RETURN_OLD, RETURN_NEW, 1
)

# Validate the complete modified module before writing.
compile(
    patched,
    str(TARGET),
    "exec"
)

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

backup = BACKUP_DIR / (
    f"report_generator_before_pose_review_{stamp}.py"
)

backup.write_text(source, encoding="utf-8")

temp = TARGET.with_name(
    TARGET.name + ".tmp"
)

temp.write_text(patched, encoding="utf-8")
os.replace(temp, TARGET)

print("=" * 60)
print("POSE REVIEW REPORT INTEGRATION")
print("=" * 60)
print("Backup:", backup)
print("Modified:", TARGET)
print("Syntax validation: PASS")
print("Decision Engine: NOT MODIFIED")
print("Agent orchestration: NOT MODIFIED")
print("Source CSV files: NOT MODIFIED")
print("Integration complete.")
