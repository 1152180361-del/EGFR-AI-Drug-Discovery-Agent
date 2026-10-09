from pathlib import Path
import time

import pandas as pd
import requests


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "chembl_egfr_ic50_raw.csv"

EXACT_FILE = (
    INTERIM_DIR /
    "egfr_ic50_binding_exact.csv"
)


# =========================================
# Target definition
# =========================================

TARGET_CHEMBL_ID = "CHEMBL203"

BASE_URL = (
    "https://www.ebi.ac.uk/"
    "chembl/api/data/activity.json"
)


# =========================================
# Query settings
# =========================================

PAGE_SIZE = 200

params = {
    "target_chembl_id": TARGET_CHEMBL_ID,
    "standard_type": "IC50",
    "assay_type": "B",
    "limit": PAGE_SIZE,
    "offset": 0,
}


# =========================================
# HTTP session
# =========================================

session = requests.Session()

session.headers.update({
    "User-Agent": "EGFR-AIDD-Pipeline/1.0"
})


# =========================================
# Request with retry
# =========================================

def request_page(query_params, max_retries=5):

    for attempt in range(1, max_retries + 1):

        try:

            response = session.get(
                BASE_URL,
                params=query_params,
                timeout=60
            )

            if response.status_code == 200:
                return response

            print(
                f"HTTP {response.status_code} "
                f"(attempt {attempt}/{max_retries})"
            )

        except requests.RequestException as error:

            print(
                f"Request error "
                f"(attempt {attempt}/{max_retries}): "
                f"{error}"
            )

        wait_time = 2 ** attempt

        print(
            f"Waiting {wait_time} seconds..."
        )

        time.sleep(wait_time)

    raise RuntimeError(
        "ChEMBL API failed after multiple retries."
    )


# =========================================
# Download
# =========================================

print("Querying ChEMBL REST API...")
print("Target:", TARGET_CHEMBL_ID)
print("Endpoint: IC50")
print("Assay type: B")

all_records = []

while True:

    print(
        f"\nDownloading offset "
        f"{params['offset']} ..."
    )

    response = request_page(params)

    payload = response.json()

    records = payload.get(
        "activities",
        []
    )

    page_meta = payload.get(
        "page_meta",
        {}
    )

    total_count = page_meta.get(
        "total_count"
    )

    all_records.extend(records)

    print(
        "Downloaded this page:",
        len(records)
    )

    print(
        "Downloaded total:",
        len(all_records)
    )

    if total_count is not None:

        print(
            "ChEMBL total count:",
            total_count
        )

    if len(records) == 0:
        break

    if (
        total_count is not None
        and len(all_records) >= total_count
    ):
        break

    if len(records) < PAGE_SIZE:
        break

    params["offset"] += len(records)

    time.sleep(0.5)


# =========================================
# Convert to DataFrame
# =========================================

df_raw = pd.json_normalize(
    all_records
)


# =========================================
# Save untouched raw query result
# =========================================

df_raw.to_csv(
    RAW_FILE,
    index=False
)

print("\n=================================")
print("RAW ChEMBL EGFR dataset")
print("=================================")

print("Rows:", len(df_raw))

if not df_raw.empty:

    if "molecule_chembl_id" in df_raw.columns:
        print(
            "Unique compounds:",
            df_raw[
                "molecule_chembl_id"
            ].nunique()
        )

    if "assay_chembl_id" in df_raw.columns:
        print(
            "Unique assays:",
            df_raw[
                "assay_chembl_id"
            ].nunique()
        )

    if "standard_relation" in df_raw.columns:

        print("\nStandard relations:")

        print(
            df_raw[
                "standard_relation"
            ].value_counts(
                dropna=False
            )
        )

    if "standard_units" in df_raw.columns:

        print("\nStandard units:")

        print(
            df_raw[
                "standard_units"
            ].value_counts(
                dropna=False
            )
        )

    if "target_organism" in df_raw.columns:

        print("\nTarget organisms:")

        print(
            df_raw[
                "target_organism"
            ].value_counts(
                dropna=False
            )
        )


# =========================================
# Local exact-relation filtering
# =========================================

df_exact = df_raw.copy()

if "standard_relation" in df_exact.columns:

    df_exact = df_exact[
        df_exact["standard_relation"] == "="
    ].copy()


# =========================================
# Save interim exact IC50 dataset
# =========================================

df_exact.to_csv(
    EXACT_FILE,
    index=False
)

print("\n=================================")
print("Exact IC50 binding subset")
print("=================================")

print("Rows:", len(df_exact))

if "molecule_chembl_id" in df_exact.columns:

    print(
        "Unique compounds:",
        df_exact[
            "molecule_chembl_id"
        ].nunique()
    )

if "assay_chembl_id" in df_exact.columns:

    print(
        "Unique assays:",
        df_exact[
            "assay_chembl_id"
        ].nunique()
    )

print("\nSaved RAW file:")
print(RAW_FILE)

print("\nSaved exact subset:")
print(EXACT_FILE)
