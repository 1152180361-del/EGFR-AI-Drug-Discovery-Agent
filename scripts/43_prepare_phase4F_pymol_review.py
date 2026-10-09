from pathlib import Path
import pandas as pd
import json


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

SHORTLIST = (
    ROOT
    / "results/zinc15_screening/"
      "phase4E_visual_review_shortlist.csv"
)

FEATURES = (
    ROOT
    / "docking/3W32/phase4C/reports/"
      "phase4C_top1_pose_features.csv"
)

RECEPTOR = (
    ROOT
    / "docking/3W32/receptor/"
      "3W32_clean_receptor.pdb"
)

REFERENCE_LIGAND = (
    ROOT
    / "docking/3W32/ligand/"
      "W32_3W32_crystal.pdb"
)

OUT_DIR = (
    ROOT
    / "docking/3W32/"
      "phase4F_visual_review"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKLIST_OUT = (
    OUT_DIR
    / "phase4F_visual_review_checklist.csv"
)

REPORT_OUT = (
    OUT_DIR
    / "phase4F_visual_review_manifest.json"
)

MASTER_PML = (
    OUT_DIR
    / "phase4F_all9_overview.pml"
)


# ============================================================
# Helpers
# ============================================================

def extract_top1_pdbqt(
    source,
    destination,
):

    lines_out = []

    saw_model = False
    in_model1 = False

    with open(
        source,
        errors="replace",
    ) as f:

        for line in f:

            if line.startswith("MODEL"):

                saw_model = True

                try:
                    model_num = int(
                        line.split()[1]
                    )
                except Exception:
                    model_num = 1

                if model_num == 1:
                    in_model1 = True
                    continue

                if in_model1:
                    break

                continue


            if line.startswith("ENDMDL"):

                if in_model1:
                    break

                continue


            if (
                saw_model
                and not in_model1
            ):
                continue


            if line.startswith(
                ("ATOM", "HETATM")
            ):

                lines_out.append(
                    line
                )


    if not lines_out:

        raise RuntimeError(
            f"No Top-1 atoms extracted from {source}"
        )


    destination.write_text(
        "".join(
            lines_out
        )
    )


def q(path):

    return str(
        path
    ).replace(
        "\\",
        "/",
    )


# ============================================================
# Load
# ============================================================

shortlist = pd.read_csv(
    SHORTLIST
)

features = pd.read_csv(
    FEATURES
)


df = shortlist.merge(
    features[
        [
            "phase4C_candidate_id",
            "docked_pdbqt",
        ]
    ],
    on="phase4C_candidate_id",
    how="left",
    suffixes=(
        "",
        "_features",
    ),
)


print(
    "\n========================================"
)

print(
    "PHASE 4F PYMOL VISUAL REVIEW PREPARATION"
)

print(
    "========================================"
)

print(
    "\nCandidates:",
    len(df)
)


# ============================================================
# Checklist
# ============================================================

checklist_rows = []

manifest_candidates = []


# ============================================================
# Master PyMOL script
# ============================================================

master_lines = [
    "reinitialize",
    "bg_color white",
    f'load {q(RECEPTOR)}, receptor',
    f'load {q(REFERENCE_LIGAND)}, W32_reference',
    "hide everything",
    "show cartoon, receptor",
    "color gray80, receptor",
    "show sticks, W32_reference",
    "color yellow, W32_reference",
    "set stick_radius, 0.18",
    "set cartoon_transparency, 0.35",
]


for _, row in df.iterrows():

    cid = str(
        row[
            "phase4C_candidate_id"
        ]
    )

    category = str(
        row[
            "final_evidence_category"
        ]
    )

    zinc_id = str(
        row[
            "zinc_id"
        ]
    )

    docked = Path(
        row[
            "docked_pdbqt"
        ]
    )


    candidate_dir = (
        OUT_DIR
        / cid
    )

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    top1_file = (
        candidate_dir
        / f"{cid}_top1.pdbqt"
    )


    extract_top1_pdbqt(
        docked,
        top1_file,
    )


    # --------------------------------------------------------
    # PyMOL object names
    # --------------------------------------------------------

    obj = (
        "lig_"
        + cid
    )

    pocket = (
        "pocket_"
        + cid
    )

    hinge = (
        "hinge_"
        + cid
    )


    # --------------------------------------------------------
    # Per-candidate PML
    # --------------------------------------------------------

    pml_path = (
        candidate_dir
        / f"{cid}_review.pml"
    )

    pse_path = (
        candidate_dir
        / f"{cid}_review.pse"
    )


    if category.startswith(
        "SUPPORTED"
    ):

        ligand_color = "cyan"

    else:

        ligand_color = "magenta"


    pml = [
        "reinitialize",
        "bg_color white",

        f'load {q(RECEPTOR)}, receptor',
        f'load {q(REFERENCE_LIGAND)}, W32_reference',
        f'load {q(top1_file)}, {obj}',

        "hide everything",

        "show cartoon, receptor",
        "color gray80, receptor",
        "set cartoon_transparency, 0.45",

        "show sticks, W32_reference",
        "color yellow, W32_reference",

        f"show sticks, {obj}",
        f"color {ligand_color}, {obj}",

        (
            f"select {pocket}, "
            f"byres (receptor within 5.0 of {obj})"
        ),

        f"show sticks, {pocket}",
        f"color wheat, {pocket}",

        (
            f"select {hinge}, "
            "receptor and chain A "
            "and resi 791+792+793+794+796+797"
        ),

        f"show sticks, {hinge}",
        f"color tv_orange, {hinge}",

        (
            "select key_ATP, "
            "receptor and chain A and "
            "resi 718+719+720+743+744+745+"
            "766+788+789+790+791+792+793+"
            "794+796+797+844+853+854+855+856"
        ),

        "show sticks, key_ATP",

        # MET793 proximity — geometric only
        (
            f"distance {cid}_MET793_N, "
            "receptor and chain A and resi 793 and name N, "
            f"{obj} and elem N+O+S, 3.5"
        ),

        (
            f"distance {cid}_MET793_O, "
            "receptor and chain A and resi 793 and name O, "
            f"{obj} and elem N+O+S, 3.5"
        ),

        # W32 reference hinge geometry
        (
            "distance W32_MET793_N, "
            "receptor and chain A and resi 793 and name N, "
            "W32_reference and elem N+O+S, 3.5"
        ),

        (
            "distance W32_MET793_O, "
            "receptor and chain A and resi 793 and name O, "
            "W32_reference and elem N+O+S, 3.5"
        ),

        "set dash_width, 2.0",
        "set dash_gap, 0.25",
        "set label_size, 16",

        (
            f"zoom ({obj} or W32_reference "
            f"or {pocket}), 7"
        ),

        "set ray_opaque_background, off",

        f'save {q(pse_path)}',

        (
            f'png {q(candidate_dir / (cid + "_review.png"))}, '
            "1600, 1200, ray=1"
        ),
    ]


    pml_path.write_text(
        "\n".join(
            pml
        )
        + "\n"
    )


    # --------------------------------------------------------
    # Master overview
    # --------------------------------------------------------

    master_lines.append(
        f'load {q(top1_file)}, {obj}'
    )

    master_lines.append(
        f"show sticks, {obj}"
    )

    master_lines.append(
        f"color {ligand_color}, {obj}"
    )


    # --------------------------------------------------------
    # Checklist row
    # --------------------------------------------------------

    checklist_rows.append(
        {
            "visual_review_order":
                row[
                    "visual_review_order"
                ],

            "phase4C_candidate_id":
                cid,

            "zinc_id":
                zinc_id,

            "evidence_category":
                category,

            "qsar_predicted_pIC50":
                row[
                    "qsar_predicted_pIC50"
                ],

            "max_training_tanimoto":
                row[
                    "max_training_tanimoto"
                ],

            "rf_tree_sd_pIC50":
                row[
                    "rf_tree_sd_pIC50"
                ],

            "reference_contact_jaccard":
                row[
                    "reference_contact_jaccard"
                ],

            "key_ATP_contact_coverage":
                row[
                    "key_ATP_contact_coverage"
                ],

            "MET793_N_distance_A":
                row[
                    "MET793_N_hetero_distance_A"
                ],

            "MET793_O_distance_A":
                row[
                    "MET793_O_hetero_distance_A"
                ],

            # Manual visual-review fields
            "ATP_pocket_occupation":
                "",

            "hinge_orientation_plausible":
                "",

            "MET793_interaction_plausible":
                "",

            "alternative_hinge_interaction":
                "",

            "buried_polar_group_concern":
                "",

            "steric_geometry_concern":
                "",

            "W32_overlay_plausible":
                "",

            "visual_review_decision":
                "",

            "visual_review_notes":
                "",

            "review_pml":
                str(
                    pml_path
                ),

            "review_pse":
                str(
                    pse_path
                ),
        }
    )


    manifest_candidates.append(
        {
            "candidate_id":
                cid,

            "zinc_id":
                zinc_id,

            "category":
                category,

            "top1_pose":
                str(
                    top1_file
                ),

            "review_pml":
                str(
                    pml_path
                ),

            "expected_pse":
                str(
                    pse_path
                ),
        }
    )


# ============================================================
# Finish master overview
# ============================================================

master_lines += [
    "zoom W32_reference, 10",
    "set ray_opaque_background, off",
    (
        f'save '
        f'{q(OUT_DIR / "phase4F_all9_overview.pse")}'
    ),
]


MASTER_PML.write_text(
    "\n".join(
        master_lines
    )
    + "\n"
)


# ============================================================
# Save checklist
# ============================================================

checklist = pd.DataFrame(
    checklist_rows
)


checklist = checklist.sort_values(
    "visual_review_order"
)


checklist.to_csv(
    CHECKLIST_OUT,
    index=False,
)


# ============================================================
# Manifest
# ============================================================

manifest = {
    "n_candidates":
        int(
            len(df)
        ),

    "receptor":
        str(
            RECEPTOR
        ),

    "reference_ligand":
        str(
            REFERENCE_LIGAND
        ),

    "master_pml":
        str(
            MASTER_PML
        ),

    "checklist":
        str(
            CHECKLIST_OUT
        ),

    "candidates":
        manifest_candidates,

    "color_scheme": {
        "W32_reference":
            "yellow",

        "supported_candidates":
            "cyan",

        "exploratory_candidates":
            "magenta",

        "hinge_region":
            "orange",
    },

    "manual_review_principle":
        (
            "Visual review is qualitative structural "
            "assessment only. It does not create a "
            "new potency score."
        ),
}


REPORT_OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
)


# ============================================================
# Terminal output
# ============================================================

print(
    "\n========================================"
)

print(
    "PHASE 4F VISUAL REVIEW FILES"
)

print(
    "========================================"
)


print(
    "\nCandidates prepared:",
    len(
        checklist
    )
)


print(
    "\nReview order:"
)


print(
    checklist[
        [
            "visual_review_order",
            "phase4C_candidate_id",
            "zinc_id",
            "evidence_category",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nMaster overview:"
)

print(
    MASTER_PML
)


print(
    "\nChecklist:"
)

print(
    CHECKLIST_OUT
)


print(
    "\n========================================"
)

print(
    "PHASE 4F PREPARATION: PASS"
)

print(
    "========================================"
)
