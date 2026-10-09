from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
from rdkit import Chem

from egfr_agent.tools.structure_tool import StructureTool


ROOT = Path(__file__).resolve().parents[1]

SOURCE_CSV = (
    ROOT / "results/zinc15_screening"
    / "phase4E_final_candidate_evidence_matrix.csv"
)

RECEPTOR = (
    ROOT / "docking/3W32/receptor"
    / "3W32_clean_receptor.pdb"
)

REFERENCE = (
    ROOT / "docking/3W32/ligand"
    / "W32_3W32_crystal.pdb"
)

OUTPUT = ROOT / "reports/pose_review_geometry"

CANDIDATES = [
    "EGFR_CAND_003",
    "EGFR_CAND_020",
]

ATOM_TYPES = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "O": "O",
    "OA": "O",
    "S": "S",
    "SA": "S",
    "P": "P",
    "F": "F",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "H": "H",
    "HD": "H",
    "HS": "H",
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def xyz_from_line(line):
    return np.array(
        [
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ],
        dtype=float,
    )


def read_pdb(path, records):

    atoms = []

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            if line[:6].strip() not in records:
                continue

            # Avoid duplicated alternate locations.
            if line[16:17] not in (" ", "A"):
                continue

            element = (
                line[76:78].strip().upper()
                if len(line) >= 78
                else ""
            )

            if not element:
                name = line[12:16].strip()
                element = re.sub(
                    r"^[0-9]+", "", name
                )[:1].upper()

            if element in {"H", "D"}:
                continue

            atoms.append({
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "resi": line[22:26].strip(),
                "element": element,
                "xyz": xyz_from_line(line),
            })

    if not atoms:
        raise ValueError(
            f"No heavy atoms in {path}"
        )

    return atoms


def read_pdbqt_top1(path):

    atoms = []
    score = None
    model_started = False

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            if line.startswith("MODEL"):

                if model_started:
                    break

                model_started = True
                continue

            if line.startswith("ENDMDL"):
                break

            if "VINA RESULT:" in line:

                match = re.search(
                    r"VINA RESULT:\s*(-?\d+(?:\.\d+)?)",
                    line,
                )

                if match and score is None:
                    score = float(match.group(1))

            if line[:6].strip() not in {
                "ATOM", "HETATM"
            }:
                continue

            atom_type = line.split()[-1].upper()

            if atom_type not in ATOM_TYPES:
                raise ValueError(
                    "Unrecognized PDBQT atom type: "
                    + atom_type
                )

            element = ATOM_TYPES[atom_type]

            if element == "H":
                continue

            atoms.append({
                "name": line[12:16].strip(),
                "element": element,
                "xyz": xyz_from_line(line),
            })

    if not atoms:
        raise ValueError(
            f"No ligand heavy atoms in {path}"
        )

    return atoms, score


def array(atoms):

    return np.array(
        [atom["xyz"] for atom in atoms],
        dtype=float,
    )


def nearest_met_backbone(
    receptor,
    ligand,
    backbone_atom,
):

    met = [
        atom
        for atom in receptor
        if (
            atom["chain"] == "A"
            and atom["resi"] == "793"
            and atom["resname"] == "MET"
            and atom["name"] == backbone_atom
        )
    ]

    if len(met) != 1:
        raise ValueError(
            "Expected one MET793 backbone "
            f"{backbone_atom}; found {len(met)}"
        )

    ligand_no = [
        atom
        for atom in ligand
        if atom["element"] in {"N", "O"}
    ]

    if not ligand_no:
        raise ValueError(
            "Ligand has no N/O heavy atoms."
        )

    distances = [
        (
            float(
                np.linalg.norm(
                    atom["xyz"] - met[0]["xyz"]
                )
            ),
            atom["name"],
            atom["element"],
        )
        for atom in ligand_no
    ]

    return min(
        distances,
        key=lambda value: value[0],
    )


def write_ligand_pdb(path, ligand):

    with path.open("w") as handle:

        handle.write(
            "REMARK COORDINATE-ONLY PDBQT CONVERSION\n"
        )

        handle.write(
            "REMARK BOND ORDER AND HYDROGEN BONDS "
            "NOT VALIDATED\n"
        )

        for index, atom in enumerate(
            ligand, 1
        ):

            x, y, z = atom["xyz"]

            name = atom["name"][:4]
            element = atom["element"]

            handle.write(
                f"HETATM{index:5d} "
                f"{name:<4s} "
                f"LIG L   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}"
                f"          {element:>2s}\n"
            )

        handle.write("END\n")


def write_pymol_script(
    path,
    ligand_file,
):

    commands = [
        "reinitialize",
        f"load {RECEPTOR}, receptor",
        f"load {REFERENCE}, W32_reference",
        f"load {ligand_file}, ligand",
        "hide everything",
        "bg_color white",
        "show cartoon, receptor",
        "color gray70, receptor",
        "set cartoon_transparency, 0.35",
        "select pocket, byres "
        "(receptor within 5 of ligand)",
        "show sticks, pocket",
        "color gray60, pocket and elem C",
        "select MET793, receptor and "
        "chain A and resi 793",
        "show sticks, MET793",
        "color yellow, MET793 and elem C",
        "show spheres, ligand",
        "set sphere_scale, 0.20, ligand",
        "color cyan, ligand and elem C",
        "color blue, ligand and elem N",
        "color red, ligand and elem O",
        "disable W32_reference",
        "distance MET793_proximity, "
        "(MET793 and name N+O), "
        "(ligand and elem N+O), 4.0",
        "set dash_color, orange",
        "set dash_radius, 0.06",
        "zoom ligand, 10",
        "orient ligand",
    ]

    path.write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )


def main():

    for path in [
        SOURCE_CSV,
        RECEPTOR,
        REFERENCE,
    ]:

        if not path.is_file():
            raise FileNotFoundError(path)

    # Source 1: original Phase 4E matrix.
    # Contains SMILES, ZINC IDs and PDBQT paths.
    source = pd.read_csv(SOURCE_CSV)

    key = "phase4C_candidate_id"

    if source[key].duplicated().any():
        raise ValueError(
            "Duplicate IDs in original evidence."
        )

    source = source.set_index(key)

    # Source 2: integrated StructureTool evidence.
    # Contains MET793_distance.
    integrated = StructureTool().run(
        top_n=100000
    )

    if "MET793_distance" not in integrated.columns:
        raise ValueError(
            "StructureTool evidence is missing "
            "MET793_distance."
        )

    if integrated[key].duplicated().any():
        raise ValueError(
            "Duplicate IDs in integrated evidence."
        )

    integrated = integrated.set_index(key)

    receptor = read_pdb(
        RECEPTOR,
        {"ATOM"},
    )

    reference = read_pdb(
        REFERENCE,
        {"ATOM", "HETATM"},
    )

    receptor_xyz = array(receptor)

    reference_center = array(
        reference
    ).mean(axis=0)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for cid in CANDIDATES:

        if cid not in source.index:
            raise ValueError(
                f"Missing Phase 4E candidate: {cid}"
            )

        if cid not in integrated.index:
            raise ValueError(
                f"Missing integrated candidate: {cid}"
            )

        row = source.loc[cid]
        combined = integrated.loc[cid]

        pose_path = Path(
            str(row["docked_pdbqt"])
        )

        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)

        ligand, vina_score = read_pdbqt_top1(
            pose_path
        )

        smiles = str(
            row["standardized_smiles"]
        )

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            raise ValueError(
                f"Invalid SMILES: {cid}"
            )

        smiles_heavy = mol.GetNumHeavyAtoms()
        pose_heavy = len(ligand)

        if smiles_heavy != pose_heavy:
            raise ValueError(
                f"{cid}: SMILES heavy count "
                f"{smiles_heavy}, PDBQT heavy "
                f"count {pose_heavy}. "
                "Atom mapping requires review."
            )

        ligand_xyz = array(ligand)

        distances = np.linalg.norm(
            ligand_xyz[:, None, :]
            - receptor_xyz[None, :, :],
            axis=2,
        )

        clashes = int(
            (distances < 1.5).sum()
        )

        close_pairs = int(
            (distances < 2.0).sum()
        )

        contact_indexes = np.where(
            distances < 5.0
        )[1]

        contacts = sorted({
            (
                receptor[int(i)]["chain"]
                + ":"
                + receptor[int(i)]["resname"]
                + receptor[int(i)]["resi"]
            )
            for i in contact_indexes
        })

        met_n = nearest_met_backbone(
            receptor,
            ligand,
            "N",
        )

        met_o = nearest_met_backbone(
            receptor,
            ligand,
            "O",
        )

        calculated_mean = (
            met_n[0] + met_o[0]
        ) / 2.0

        previous_metric = float(
            combined["MET793_distance"]
        )

        difference = (
            calculated_mean - previous_metric
        )

        center_distance = float(
            np.linalg.norm(
                ligand_xyz.mean(axis=0)
                - reference_center
            )
        )

        ligand_file = (
            OUTPUT
            / f"{cid}_top1_coordinates_only.pdb"
        )

        pml_file = (
            OUTPUT
            / f"{cid}_inspect.pml"
        )

        write_ligand_pdb(
            ligand_file,
            ligand,
        )

        write_pymol_script(
            pml_file,
            ligand_file,
        )

        results.append({
            "candidate_id": cid,
            "zinc_id": row["zinc_id"],
            "docked_pdbqt": str(pose_path),
            "docked_pdbqt_sha256": sha256(
                pose_path
            ),
            "smiles_heavy_atoms": smiles_heavy,
            "pdbqt_heavy_atoms": pose_heavy,
            "heavy_atom_count_match": True,
            "vina_top1_score": vina_score,
            "minimum_protein_distance_A": float(
                distances.min()
            ),
            "clash_pairs_lt1p5A": clashes,
            "close_pairs_lt2p0A": close_pairs,
            "contact_residue_count_5A": len(
                contacts
            ),
            "contact_residues_5A": ";".join(
                contacts
            ),
            "MET793_N_min_NO_A": met_n[0],
            "MET793_N_nearest_ligand_atom": (
                met_n[1]
            ),
            "MET793_O_min_NO_A": met_o[0],
            "MET793_O_nearest_ligand_atom": (
                met_o[1]
            ),
            "MET793_mean_two_minima_A": (
                calculated_mean
            ),
            "MET793_integrated_metric_A": (
                previous_metric
            ),
            "MET793_metric_difference_A": (
                difference
            ),
            "ligand_to_W32_centroid_A": (
                center_distance
            ),
            "hydrogen_bond_status": (
                "not_assessed"
            ),
            "manual_review_status": (
                "not_started"
            ),
            "pymol_script": str(pml_file),
        })

    frame = pd.DataFrame(results)

    output_csv = (
        OUTPUT
        / "CAND003_CAND020_geometry_audit.csv"
    )

    frame.to_csv(
        output_csv,
        index=False,
    )

    print("=" * 65)
    print("EGFR POSE GEOMETRY AUDIT")
    print("=" * 65)

    columns = [
        "candidate_id",
        "pdbqt_heavy_atoms",
        "clash_pairs_lt1p5A",
        "MET793_N_min_NO_A",
        "MET793_O_min_NO_A",
        "MET793_mean_two_minima_A",
        "MET793_integrated_metric_A",
        "MET793_metric_difference_A",
    ]

    print(
        frame[columns].to_string(
            index=False
        )
    )

    print("\nGeometry audit saved:")
    print(output_csv)

    print(
        "\nNOTE: Different MET793 distance "
        "definitions can produce different values."
    )

    print(
        "No hydrogen bond has been "
        "automatically confirmed."
    )


if __name__ == "__main__":
    main()
