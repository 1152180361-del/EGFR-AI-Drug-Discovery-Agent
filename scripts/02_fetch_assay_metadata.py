from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_ic50_binding_exact.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_assay_metadata.csv"
)

CHECKPOINT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_assay_metadata_checkpoint.csv"
)


# =========================================
# Read unique assay IDs only
# =========================================

df = pd.read_csv(
    INPUT_FILE,
    usecols=["assay_chembl_id"],
    dtype={"assay_chembl_id": "string"},
    low_memory=False,
)

assay_ids = (
    df["assay_chembl_id"]
    .dropna()
    .drop_duplicates()
    .tolist()
)

print("Unique assay IDs:", len(assay_ids))


# =========================================
# Resume from checkpoint if available
# =========================================

metadata_records = []
completed_ids = set()

if CHECKPOINT_FILE.exists():

    checkpoint_df = pd.read_csv(
        CHECKPOINT_FILE,
        low_memory=False
    )

    metadata_records = (
        checkpoint_df
        .to_dict("records")
    )

    if "assay_chembl_id" in checkpoint_df.columns:

        completed_ids = set(
            checkpoint_df[
                "assay_chembl_id"
            ]
            .dropna()
            .astype(str)
        )

    print(
        "Checkpoint found."
    )

    print(
        "Already completed:",
        len(completed_ids)
    )


remaining_ids = [
    assay_id
    for assay_id in assay_ids
    if assay_id not in completed_ids
]

print(
    "Remaining assays:",
    len(remaining_ids)
)


# =========================================
# HTTP session
# =========================================

session = requests.Session()

session.headers.update({
    "User-Agent":
    "EGFR-AIDD-Pipeline/1.0"
})


# =========================================
# Fetch one assay
# =========================================

def fetch_assay_metadata(
    assay_id,
    max_retries=5
):

    url = (
        "https://www.ebi.ac.uk/"
        "chembl/api/data/"
        f"assay/{assay_id}.json"
    )

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            response = session.get(
                url,
                timeout=60
            )

            if response.status_code == 200:

                data = response.json()

                return {
                    "assay_chembl_id":
                        data.get(
                            "assay_chembl_id"
                        ),

                    "assay_type":
                        data.get(
                            "assay_type"
                        ),

                    "assay_type_description":
                        data.get(
                            "assay_type_description"
                        ),

                    "description":
                        data.get(
                            "description"
                        ),

                    "confidence_score":
                        data.get(
                            "confidence_score"
                        ),

                    "confidence_description":
                        data.get(
                            "confidence_description"
                        ),

                    "relationship_type":
                        data.get(
                            "relationship_type"
                        ),

                    "relationship_description":
                        data.get(
                            "relationship_description"
                        ),

                    "target_chembl_id":
                        data.get(
                            "target_chembl_id"
                        ),

                    "assay_organism":
                        data.get(
                            "assay_organism"
                        ),

                    "assay_cell_type":
                        data.get(
                            "assay_cell_type"
                        ),

                    "assay_tissue":
                        data.get(
                            "assay_tissue"
                        ),

                    "assay_subcellular_fraction":
                        data.get(
                            "assay_subcellular_fraction"
                        ),

                    "bao_format":
                        data.get(
                            "bao_format"
                        ),

                    "bao_label":
                        data.get(
                            "bao_label"
                        ),

                    "variant_sequence":
                        data.get(
                            "variant_sequence"
                        ),

                    "document_chembl_id":
                        data.get(
                            "document_chembl_id"
                        ),
                }

            print(
                f"\n{assay_id}: "
                f"HTTP "
                f"{response.status_code}"
            )

        except requests.RequestException as error:

            print(
                f"\n{assay_id}: "
                f"{error}"
            )

        wait_time = 2 ** attempt

        time.sleep(
            wait_time
        )

    return {
        "assay_chembl_id":
            assay_id,

        "fetch_failed":
            True
    }


# =========================================
# Download metadata
# =========================================

for i, assay_id in enumerate(
    tqdm(
        remaining_ids,
        desc="Fetching assays"
    ),
    start=1
):

    record = fetch_assay_metadata(
        assay_id
    )

    metadata_records.append(
        record
    )

    # Checkpoint every 50 assays
    if i % 50 == 0:

        pd.DataFrame(
            metadata_records
        ).to_csv(
            CHECKPOINT_FILE,
            index=False
        )

    time.sleep(0.1)


# =========================================
# Final dataset
# =========================================

metadata_df = pd.DataFrame(
    metadata_records
)

metadata_df = (
    metadata_df
    .drop_duplicates(
        subset=[
            "assay_chembl_id"
        ],
        keep="last"
    )
    .sort_values(
        "assay_chembl_id"
    )
    .reset_index(
        drop=True
    )
)

metadata_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================
# QC summary
# =========================================

print("\n=================================")
print("Assay metadata summary")
print("=================================")

print(
    "Assays:",
    len(metadata_df)
)


if "confidence_score" in metadata_df.columns:

    print(
        "\nConfidence scores:"
    )

    print(
        metadata_df[
            "confidence_score"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


if "bao_label" in metadata_df.columns:

    print(
        "\nBAO formats:"
    )

    print(
        metadata_df[
            "bao_label"
        ]
        .value_counts(
            dropna=False
        )
        .head(20)
    )


if "assay_cell_type" in metadata_df.columns:

    print(
        "\nAssays with cell type:",
        metadata_df[
            "assay_cell_type"
        ]
        .notna()
        .sum()
    )


if "variant_sequence" in metadata_df.columns:

    print(
        "Assays with variant sequence:",
        metadata_df[
            "variant_sequence"
        ]
        .notna()
        .sum()
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)
