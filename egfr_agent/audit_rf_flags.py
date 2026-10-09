from pathlib import Path

import pandas as pd

from .tools.structure_tool import StructureTool
from .risk_engine import EGFRAgentRiskEngine


BASE = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    BASE
    / "results/zinc15_screening"
    / "phase4E_final_candidate_evidence_matrix.csv"
)

OUTPUT_FILE = (
    BASE
    / "reports"
    / "EGFR_RF_Flag_Definition_Audit.csv"
)


def parse_flags(series):

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if values.isna().any():
        raise ValueError(
            "Invalid original RF_spread_warning values."
        )

    return values.astype(bool)


def main():

    print("=" * 65)
    print("EGFR RF FLAG DEFINITION AUDIT")
    print("=" * 65)

    data = StructureTool().run(
        top_n=100000
    )

    required = [
        "phase4C_candidate_id",
        "candidate_track",
        "rf_tree_sd_pIC50",
        "RF_spread_warning",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing evidence fields: "
            + str(missing)
        )

    key = "phase4C_candidate_id"

    if data[key].duplicated().any():
        raise ValueError(
            "Duplicate candidate identifiers."
        )

    threshold = (
        EGFRAgentRiskEngine.RF_SD_WARNING
    )

    audit = data[required].copy()

    audit["RF_spread_warning_original"] = (
        parse_flags(
            audit["RF_spread_warning"]
        )
    )

    audit["RF_spread_warning_expected"] = (
        audit["candidate_track"]
        == "SUPPORTED_B_HIGH_SPREAD"
    )

    audit["original_flag_matches_definition"] = (
        audit["RF_spread_warning_original"]
        == audit["RF_spread_warning_expected"]
    )

    audit["rf_tree_sd_pIC50"] = pd.to_numeric(
        audit["rf_tree_sd_pIC50"],
        errors="raise",
    )

    if audit["rf_tree_sd_pIC50"].isna().any():
        raise ValueError(
            "Missing RF tree SD measurements."
        )

    audit["rf_sd_above_threshold"] = (
        audit["rf_tree_sd_pIC50"]
        > threshold
    )

    # ==================================
    # Source-file consistency check
    # ==================================

    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            str(SOURCE_FILE)
        )

    source = pd.read_csv(SOURCE_FILE)

    source_required = [
        key,
        "RF_spread_warning",
    ]

    if any(
        col not in source.columns
        for col in source_required
    ):
        raise ValueError(
            "Phase 4E source fields missing."
        )

    if source[key].duplicated().any():
        raise ValueError(
            "Duplicate IDs in Phase 4E source."
        )

    source_subset = source[
        source_required
    ].copy()

    source_subset[
        "RF_spread_warning_phase4E"
    ] = parse_flags(
        source_subset["RF_spread_warning"]
    )

    source_subset = source_subset.drop(
        columns=["RF_spread_warning"]
    )

    audit = audit.merge(
        source_subset,
        on=key,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if audit["_merge"].ne("both").any():
        raise ValueError(
            "Some candidates are absent "
            "from Phase 4E source."
        )

    audit = audit.drop(
        columns=["_merge"]
    )

    audit["source_flag_consistent"] = (
        audit["RF_spread_warning_original"]
        == audit["RF_spread_warning_phase4E"]
    )

    # ==================================
    # Results
    # ==================================

    total = len(audit)

    category_correct = int(
        audit[
            "original_flag_matches_definition"
        ].sum()
    )

    source_correct = int(
        audit["source_flag_consistent"].sum()
    )

    original_true = int(
        audit[
            "RF_spread_warning_original"
        ].sum()
    )

    above_threshold = int(
        audit["rf_sd_above_threshold"].sum()
    )

    print("\nCandidates:", total)

    print(
        "Original category flag True:",
        original_true,
    )

    print(
        "RF SD above threshold:",
        above_threshold,
    )

    print(
        "Operational RF SD threshold:",
        threshold,
    )

    print(
        "Original category-definition checks:",
        f"{category_correct}/{total}",
    )

    print(
        "Phase 4E source checks:",
        f"{source_correct}/{total}",
    )

    print("\nCATEGORY DISTRIBUTION")

    print(
        pd.crosstab(
            audit["candidate_track"],
            audit["RF_spread_warning_original"],
        ).to_string()
    )

    print("\nNUMERICAL RF SD DISTRIBUTION")

    print(
        pd.crosstab(
            audit["candidate_track"],
            audit["rf_sd_above_threshold"],
        ).to_string()
    )

    print("\nEXAMPLE CANDIDATES")

    examples = audit.loc[
        audit[key].isin([
            "EGFR_CAND_003",
            "EGFR_CAND_020",
        ])
    ]

    print(
        examples[
            [
                key,
                "candidate_track",
                "rf_tree_sd_pIC50",
                "RF_spread_warning_original",
                "rf_sd_above_threshold",
                "original_flag_matches_definition",
            ]
        ].to_string(index=False)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nAudit saved:", OUTPUT_FILE)

    if (
        category_correct != total
        or source_correct != total
    ):
        raise RuntimeError(
            "RF source-definition audit failed."
        )

    print(
        "\nAUDIT PASS: Original source flags "
        "match the original category-based rule."
    )

    print(
        "RF SD threshold flags are separate "
        "numerical descriptors."
    )


if __name__ == "__main__":
    main()
