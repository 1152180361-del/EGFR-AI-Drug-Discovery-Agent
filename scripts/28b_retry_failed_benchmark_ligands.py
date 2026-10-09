from pathlib import Path
import subprocess
import pandas as pd

ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

BASE = ROOT / "docking/3W32/benchmark"

MANIFEST = (
    BASE
    / "benchmark_ligand_preparation_manifest.csv"
)

SDF_DIR = BASE / "sdf"
PDBQT_DIR = BASE / "pdbqt"
LOG_DIR = BASE / "logs"

TIMEOUT_SECONDS = 120


df = pd.read_csv(
    MANIFEST
)

failed_idx = df.index[
    df["status"] != "PASS"
].tolist()


print(
    "Failed molecules to retry:",
    len(failed_idx)
)


for idx in failed_idx:

    bid = str(
        df.loc[
            idx,
            "benchmark_id"
        ]
    )

    sdf = (
        SDF_DIR
        / f"{bid}.sdf"
    )

    pdbqt = (
        PDBQT_DIR
        / f"{bid}.pdbqt"
    )

    log = (
        LOG_DIR
        / f"{bid}_retry.log"
    )

    if pdbqt.exists():
        pdbqt.unlink()

    print(
        f"\nRetrying {bid} ..."
    )

    cmd = [
        "mk_prepare_ligand.py",
        "-i",
        str(sdf),
        "-o",
        str(pdbqt),
    ]

    try:

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        log.write_text(
            result.stdout
        )

        exit_code = (
            result.returncode
        )

        if (
            exit_code == 0
            and pdbqt.exists()
            and pdbqt.stat().st_size > 0
        ):

            df.loc[
                idx,
                "status"
            ] = "PASS"

            df.loc[
                idx,
                "pdbqt_created"
            ] = True

            df.loc[
                idx,
                "meeko_exit_code"
            ] = 0

            df.loc[
                idx,
                "failure_reason"
            ] = ""

            print(
                "  PASS"
            )

        else:

            df.loc[
                idx,
                "status"
            ] = "FAIL"

            df.loc[
                idx,
                "pdbqt_created"
            ] = False

            df.loc[
                idx,
                "meeko_exit_code"
            ] = exit_code

            df.loc[
                idx,
                "failure_reason"
            ] = (
                "Meeko retry failed"
            )

            print(
                "  FAIL, exit code:",
                exit_code
            )

    except subprocess.TimeoutExpired as exc:

        output = (
            exc.stdout
            if exc.stdout
            else ""
        )

        if isinstance(
            output,
            bytes
        ):
            output = output.decode(
                errors="replace"
            )

        log.write_text(
            str(output)
            + "\nTIMEOUT AFTER "
            + str(
                TIMEOUT_SECONDS
            )
            + " SECONDS\n"
        )

        df.loc[
            idx,
            "status"
        ] = "TIMEOUT"

        df.loc[
            idx,
            "pdbqt_created"
        ] = False

        df.loc[
            idx,
            "meeko_exit_code"
        ] = -999

        df.loc[
            idx,
            "failure_reason"
        ] = (
            f"Meeko timeout "
            f">{TIMEOUT_SECONDS}s"
        )

        print(
            "  TIMEOUT"
        )


df.to_csv(
    MANIFEST,
    index=False,
)


print(
    "\n========================================"
)

print(
    "FAILED LIGAND RETRY SUMMARY"
)

print(
    "========================================"
)


summary = df[
    df["status"] != "PASS"
][
    [
        "benchmark_id",
        "benchmark_label",
        "pIC50",
        "meeko_exit_code",
        "status",
        "failure_reason",
    ]
]


if len(summary) == 0:

    print(
        "\nAll 100 ligands prepared successfully."
    )

else:

    print(
        "\nStill unresolved:"
    )

    print(
        summary.to_string(
            index=False
        )
    )


print(
    "\nTotal PASS:",
    int(
        (
            df["status"]
            == "PASS"
        ).sum()
    ),
    "/",
    len(df)
)


for label in [
    "ACTIVE",
    "LOW_ACTIVE",
]:

    subset = df[
        df[
            "benchmark_label"
        ]
        == label
    ]

    n_pass = int(
        (
            subset["status"]
            == "PASS"
        ).sum()
    )

    print(
        label,
        "PASS:",
        n_pass,
        "/",
        len(subset)
    )
