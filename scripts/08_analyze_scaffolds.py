from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_qsar_modeling.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_qsar_scaffolds.csv"
)

SCAFFOLD_STATS_FILE = (
    ROOT
    / "reports"
    / "phase2_scaffold_statistics.csv"
)

TOP_SCAFFOLDS_FILE = (
    ROOT
    / "reports"
    / "phase2_top_scaffolds.csv"
)

FIGURE_FILE = (
    ROOT
    / "figures"
    / "phase2_top_scaffold_frequency.png"
)


# =========================================
# Ensure directories exist
# =========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

SCAFFOLD_STATS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Load dataset
# =========================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    "RDKit version:",
    rdBase.rdkitVersion
)

print(
    "Input QSAR compounds:",
    len(df)
)


# =========================================
# Required columns
# =========================================

required_columns = [
    "standardized_smiles",
    "standardized_inchikey",
    "pIC50",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# =========================================
# Scaffold extraction
# =========================================

results = []

failed_count = 0
acyclic_count = 0


for i, record in enumerate(
    df.to_dict("records"),
    start=1
):

    smiles = str(
        record["standardized_smiles"]
    )

    inchikey = str(
        record["standardized_inchikey"]
    )

    scaffold_smiles = None
    generic_scaffold_smiles = None

    scaffold_success = False
    is_acyclic = False
    error_message = None


    try:

        # ---------------------------------
        # Parse standardized molecule
        # ---------------------------------

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:

            raise ValueError(
                "RDKit parse failed"
            )


        # ---------------------------------
        # Bemis-Murcko scaffold
        # ---------------------------------

        scaffold_mol = (
            MurckoScaffold
            .GetScaffoldForMol(
                mol
            )
        )

        scaffold_smiles = (
            Chem.MolToSmiles(
                scaffold_mol,
                canonical=True,
                isomericSmiles=True
            )
        )


        # ---------------------------------
        # Acyclic molecule
        #
        # Bemis-Murcko scaffold can be empty
        # when no ring system exists.
        # ---------------------------------

        if scaffold_smiles == "":

            is_acyclic = True
            acyclic_count += 1

            generic_scaffold_smiles = ""


        else:

            # -----------------------------
            # Generic scaffold
            #
            # Convert atoms/bonds to a more
            # abstract scaffold representation.
            # -----------------------------

            generic_mol = (
                MurckoScaffold
                .MakeScaffoldGeneric(
                    scaffold_mol
                )
            )

            generic_scaffold_smiles = (
                Chem.MolToSmiles(
                    generic_mol,
                    canonical=True,
                    isomericSmiles=False
                )
            )


        scaffold_success = True


    except Exception as error:

        failed_count += 1

        error_message = (
            f"{type(error).__name__}: "
            f"{str(error)}"
        )


    # =====================================
    # Split scaffold ID
    #
    # Important:
    # Do NOT assign every acyclic compound
    # to one giant empty scaffold group.
    #
    # Each acyclic molecule gets its own
    # split identity for later scaffold split.
    # =====================================

    if scaffold_success:

        if is_acyclic:

            split_scaffold_id = (
                "ACYCLIC__"
                + inchikey
            )

        else:

            split_scaffold_id = (
                scaffold_smiles
            )

    else:

        split_scaffold_id = (
            "FAILED__"
            + inchikey
        )


    results.append(
        {
            "standardized_smiles":
                smiles,

            "standardized_inchikey":
                inchikey,

            "murcko_scaffold":
                scaffold_smiles,

            "generic_scaffold":
                generic_scaffold_smiles,

            "split_scaffold_id":
                split_scaffold_id,

            "is_acyclic":
                is_acyclic,

            "scaffold_success":
                scaffold_success,

            "scaffold_error":
                error_message,
        }
    )


    if i % 500 == 0:

        print(
            f"Processed "
            f"{i}/{len(df)}"
        )


# =========================================
# Build scaffold mapping
# =========================================

scaffold_df = pd.DataFrame(
    results
)


# =========================================
# Merge scaffold data back to QSAR data
# =========================================

merged = df.merge(
    scaffold_df,
    on=[
        "standardized_smiles",
        "standardized_inchikey"
    ],
    how="left",
    validate="one_to_one"
)


# =========================================
# Scaffold frequency
#
# Use Python Counter rather than pandas
# groupby.
# =========================================

scaffold_counter = Counter(
    merged[
        "split_scaffold_id"
    ].tolist()
)


# =========================================
# Add scaffold size to every compound
# =========================================

merged[
    "scaffold_size"
] = [
    scaffold_counter[
        scaffold_id
    ]
    for scaffold_id
    in merged[
        "split_scaffold_id"
    ]
]


# =========================================
# Save compound scaffold dataset
# =========================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================
# Main statistics
# =========================================

n_compounds = len(merged)

n_split_scaffolds = len(
    scaffold_counter
)

true_murcko_scaffolds = set()

generic_scaffolds = set()


for record in results:

    scaffold = record[
        "murcko_scaffold"
    ]

    generic = record[
        "generic_scaffold"
    ]

    if (
        scaffold is not None
        and scaffold != ""
    ):

        true_murcko_scaffolds.add(
            scaffold
        )

    if (
        generic is not None
        and generic != ""
    ):

        generic_scaffolds.add(
            generic
        )


n_murcko_scaffolds = len(
    true_murcko_scaffolds
)

n_generic_scaffolds = len(
    generic_scaffolds
)


# =========================================
# Scaffold size statistics
# =========================================

sizes = list(
    scaffold_counter.values()
)

singleton_scaffolds = sum(
    1
    for size in sizes
    if size == 1
)

multi_compound_scaffolds = sum(
    1
    for size in sizes
    if size > 1
)

largest_scaffold_size = max(
    sizes
)

mean_scaffold_size = (
    sum(sizes)
    / len(sizes)
)


# =========================================
# Top scaffold concentration
# =========================================

sorted_scaffolds = (
    scaffold_counter
    .most_common()
)

top_1_count = sum(
    count
    for _, count
    in sorted_scaffolds[:1]
)

top_5_count = sum(
    count
    for _, count
    in sorted_scaffolds[:5]
)

top_10_count = sum(
    count
    for _, count
    in sorted_scaffolds[:10]
)

top_20_count = sum(
    count
    for _, count
    in sorted_scaffolds[:20]
)


# =========================================
# Scaffold statistics table
# =========================================

statistics = pd.DataFrame(
    [
        [
            "QSAR compounds",
            n_compounds
        ],
        [
            "Murcko scaffolds",
            n_murcko_scaffolds
        ],
        [
            "Generic scaffolds",
            n_generic_scaffolds
        ],
        [
            "Split scaffold groups",
            n_split_scaffolds
        ],
        [
            "Acyclic compounds",
            acyclic_count
        ],
        [
            "Scaffold extraction failures",
            failed_count
        ],
        [
            "Singleton scaffold groups",
            singleton_scaffolds
        ],
        [
            "Multi-compound scaffold groups",
            multi_compound_scaffolds
        ],
        [
            "Largest scaffold size",
            largest_scaffold_size
        ],
        [
            "Mean compounds per scaffold",
            mean_scaffold_size
        ],
        [
            "Top 1 scaffold compound fraction",
            top_1_count / n_compounds
        ],
        [
            "Top 5 scaffold compound fraction",
            top_5_count / n_compounds
        ],
        [
            "Top 10 scaffold compound fraction",
            top_10_count / n_compounds
        ],
        [
            "Top 20 scaffold compound fraction",
            top_20_count / n_compounds
        ],
    ],
    columns=[
        "metric",
        "value"
    ]
)


statistics.to_csv(
    SCAFFOLD_STATS_FILE,
    index=False
)


# =========================================
# Top scaffold table
# =========================================

top_records = []


for rank, (
    scaffold_id,
    count
) in enumerate(
    sorted_scaffolds[:50],
    start=1
):

    compounds = merged[
        merged[
            "split_scaffold_id"
        ]
        == scaffold_id
    ]

    median_pic50 = (
        compounds[
            "pIC50"
        ].median()
    )

    min_pic50 = (
        compounds[
            "pIC50"
        ].min()
    )

    max_pic50 = (
        compounds[
            "pIC50"
        ].max()
    )

    top_records.append(
        {
            "rank":
                rank,

            "split_scaffold_id":
                scaffold_id,

            "n_compounds":
                count,

            "fraction":
                count
                / n_compounds,

            "median_pIC50":
                median_pic50,

            "min_pIC50":
                min_pic50,

            "max_pIC50":
                max_pic50,
        }
    )


top_df = pd.DataFrame(
    top_records
)

top_df.to_csv(
    TOP_SCAFFOLDS_FILE,
    index=False
)


# =========================================
# Plot top 20 scaffold frequencies
# =========================================

plot_df = top_df.head(
    20
).copy()

plot_df[
    "label"
] = [
    f"S{i}"
    for i in range(
        1,
        len(plot_df) + 1
    )
]


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    plot_df[
        "label"
    ],
    plot_df[
        "n_compounds"
    ]
)

plt.xlabel(
    "Top Bemis-Murcko scaffold"
)

plt.ylabel(
    "Number of compounds"
)

plt.title(
    "Top 20 EGFR Scaffold Frequencies"
)

plt.tight_layout()

plt.savefig(
    FIGURE_FILE,
    dpi=300
)

plt.close()


# =========================================
# Final summary
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 2.1 SCAFFOLD ANALYSIS"
)

print(
    "================================="
)

print(
    "QSAR compounds:",
    n_compounds
)

print(
    "Unique Murcko scaffolds:",
    n_murcko_scaffolds
)

print(
    "Unique generic scaffolds:",
    n_generic_scaffolds
)

print(
    "Split scaffold groups:",
    n_split_scaffolds
)

print(
    "Acyclic compounds:",
    acyclic_count
)

print(
    "Scaffold extraction failures:",
    failed_count
)

print(
    "\nSingleton scaffold groups:",
    singleton_scaffolds
)

print(
    "Multi-compound scaffold groups:",
    multi_compound_scaffolds
)

print(
    "Largest scaffold size:",
    largest_scaffold_size
)

print(
    "Mean compounds per scaffold:",
    round(
        mean_scaffold_size,
        3
    )
)

print(
    "\nTop 1 scaffold fraction:",
    round(
        top_1_count
        / n_compounds,
        4
    )
)

print(
    "Top 5 scaffold fraction:",
    round(
        top_5_count
        / n_compounds,
        4
    )
)

print(
    "Top 10 scaffold fraction:",
    round(
        top_10_count
        / n_compounds,
        4
    )
)

print(
    "Top 20 scaffold fraction:",
    round(
        top_20_count
        / n_compounds,
        4
    )
)


print(
    "\n--- Top 10 scaffold groups ---"
)

print(
    top_df[
        [
            "rank",
            "n_compounds",
            "fraction",
            "median_pIC50",
            "min_pIC50",
            "max_pIC50",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nSaved scaffold dataset:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSaved scaffold statistics:"
)

print(
    SCAFFOLD_STATS_FILE
)

print(
    "\nSaved top scaffold table:"
)

print(
    TOP_SCAFFOLDS_FILE
)

print(
    "\nSaved figure:"
)

print(
    FIGURE_FILE
)
