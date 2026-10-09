from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd


geometry = import_module(
    "scripts.43_audit_egfr_candidate_pose_geometry"
)

CUTOFF = 4.0

OUTPUT = (
    geometry.OUTPUT
    / "MET793_nearby_heteroatom_pairs.csv"
)


def get_pdbqt_heavy_types(path):

    atom_types = []

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            if line.startswith("ENDMDL"):
                break

            if line[:6].strip() not in {
                "ATOM", "HETATM"
            }:
                continue

            atom_type = line.split()[-1].upper()

            if atom_type not in geometry.ATOM_TYPES:
                raise ValueError(
                    "Unknown PDBQT atom type: "
                    + atom_type
                )

            if geometry.ATOM_TYPES[atom_type] == "H":
                continue

            atom_types.append(atom_type)

    return atom_types


def main():

    data = pd.read_csv(
        geometry.SOURCE_CSV
    )

    data = data.set_index(
        "phase4C_candidate_id",
        verify_integrity=True,
    )

    receptor = geometry.read_pdb(
        geometry.RECEPTOR,
        {"ATOM"},
    )

    backbone = [
        atom
        for atom in receptor
        if atom["chain"] == "A"
        and atom["resi"] == "793"
        and atom["resname"] == "MET"
        and atom["name"] in {"N", "O"}
    ]

    if sorted(
        atom["name"] for atom in backbone
    ) != ["N", "O"]:
        raise ValueError(
            "Expected MET793 backbone N and O."
        )

    records = []

    print("=" * 65)
    print("MET793 BACKBONE CONTACT ATOM TRACE")
    print("=" * 65)

    for cid in geometry.CANDIDATES:

        pose_file = Path(
            data.loc[cid, "docked_pdbqt"]
        )

        ligand, _ = geometry.read_pdbqt_top1(
            pose_file
        )

        atom_types = get_pdbqt_heavy_types(
            pose_file
        )

        if len(ligand) != len(atom_types):
            raise ValueError(
                f"{cid}: PDBQT atom types "
                "and coordinates are not aligned."
            )

        all_pairs = []

        for receptor_atom in backbone:

            for index, (ligand_atom, atom_type) in enumerate(
                zip(ligand, atom_types),
                start=1,
            ):

                if ligand_atom["element"] not in {
                    "N", "O"
                }:
                    continue

                distance = float(
                    np.linalg.norm(
                        receptor_atom["xyz"]
                        - ligand_atom["xyz"]
                    )
                )

                all_pairs.append({
                    "candidate_id": cid,
                    "receptor_residue": "A:MET793",
                    "receptor_backbone_atom": (
                        receptor_atom["name"]
                    ),
                    "ligand_heavy_atom_index": index,
                    "ligand_atom_name": (
                        ligand_atom["name"]
                    ),
                    "ligand_element": (
                        ligand_atom["element"]
                    ),
                    "ligand_pdbqt_type": atom_type,
                    "distance_A": round(distance, 4),
                    "hydrogen_bond_status": (
                        "not_assessed"
                    ),
                })

        print(f"\nCandidate: {cid}")

        for backbone_name in ("N", "O"):

            nearest = min(
                (
                    row for row in all_pairs
                    if row["receptor_backbone_atom"]
                    == backbone_name
                ),
                key=lambda row: row["distance_A"],
            )

            print(
                f"MET793 {backbone_name} nearest: "
                f"ligand atom "
                f"{nearest['ligand_heavy_atom_index']} "
                f"{nearest['ligand_atom_name']} "
                f"[{nearest['ligand_pdbqt_type']}] "
                f"distance={nearest['distance_A']:.4f} A"
            )

        near = sorted(
            (
                row for row in all_pairs
                if row["distance_A"] <= CUTOFF
            ),
            key=lambda row: row["distance_A"],
        )

        print(
            f"N/O pairs within {CUTOFF:.1f} A:",
            len(near),
        )

        for row in near:

            print(
                "  MET793 "
                + row["receptor_backbone_atom"]
                + " -- ligand "
                + str(row["ligand_heavy_atom_index"])
                + " "
                + row["ligand_atom_name"]
                + " ["
                + row["ligand_pdbqt_type"]
                + "] = "
                + f"{row['distance_A']:.4f} A"
            )

        records.extend(near)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "candidate_id",
        "receptor_residue",
        "receptor_backbone_atom",
        "ligand_heavy_atom_index",
        "ligand_atom_name",
        "ligand_element",
        "ligand_pdbqt_type",
        "distance_A",
        "hydrogen_bond_status",
    ]

    pd.DataFrame(
        records,
        columns=columns,
    ).to_csv(
        OUTPUT,
        index=False,
    )

    print("\nContact atom table saved:")
    print(OUTPUT)

    print(
        "\nThese are distance-based "
        "contacts, NOT confirmed hydrogen bonds."
    )


if __name__ == "__main__":
    main()
