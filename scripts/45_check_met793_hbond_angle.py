import json
import math
import re
from importlib import import_module
from pathlib import Path

import numpy as np


geo = import_module(
    "scripts.43_audit_egfr_candidate_pose_geometry"
)

HOPT = (
    geo.ROOT / "docking/3W32/receptor"
    / "3W32_reduce2_Hopt.pdb"
)

OUTPUT = (
    geo.OUTPUT
    / "CAND020_MET793_hbond_geometry.json"
)


def read_met793(path):
    atoms = []

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:
            if line[:6].strip() not in {
                "ATOM", "HETATM"
            }:
                continue

            if (
                line[21:22].strip() != "A"
                or line[22:26].strip() != "793"
                or line[17:20].strip() != "MET"
            ):
                continue

            if line[16:17] not in {" ", "A"}:
                continue

            name = line[12:16].strip()
            element = line[76:78].strip().upper()

            if not element:
                element = re.sub(
                    r"^\d+", "", name
                )[:1].upper()

            atoms.append({
                "name": name,
                "element": element,
                "xyz": geo.xyz_from_line(line),
            })

    return atoms


def named_atom(atoms, name):
    found = [
        atom for atom in atoms
        if atom["name"] == name
    ]

    if len(found) != 1:
        raise ValueError(
            f"Expected one MET793 {name}; "
            f"found {len(found)}"
        )

    return found[0]


def distance(a, b):
    return float(
        np.linalg.norm(a - b)
    )


def angle_at_h(donor, hydrogen, acceptor):

    first = donor - hydrogen
    second = acceptor - hydrogen

    denominator = (
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator == 0:
        raise ValueError("Zero-length angle vector.")

    cosine = float(
        np.dot(first, second) / denominator
    )

    return math.degrees(
        math.acos(
            float(np.clip(cosine, -1.0, 1.0))
        )
    )


def main():

    if not HOPT.is_file():
        raise FileNotFoundError(HOPT)

    clean_atoms = read_met793(
        geo.RECEPTOR
    )

    hopt_atoms = read_met793(
        HOPT
    )

    print("=" * 60)
    print("CAND020 MET793 HYDROGEN-BOND GEOMETRY")
    print("=" * 60)

    print("\nStep 1: receptor coordinate check")

    displacements = {}

    for name in ["N", "CA", "C", "O"]:

        clean = named_atom(
            clean_atoms, name
        )["xyz"]

        optimized = named_atom(
            hopt_atoms, name
        )["xyz"]

        displacements[name] = round(
            distance(clean, optimized), 4
        )

        print(
            f"MET793 {name}: "
            f"{displacements[name]:.4f} A"
        )

    compatible = (
        max(displacements.values()) <= 0.35
    )

    result = {
        "candidate": "EGFR_CAND_020",
        "receptor_coordinate_displacements_A": (
            displacements
        ),
        "receptor_coordinates_compatible": (
            compatible
        ),
        "hydrogen_bond_confirmed": False,
        "assessment": "not_assessed",
    }

    if not compatible:

        result["assessment"] = (
            "receptor_coordinate_mismatch"
        )

        print(
            "\nSTOP: receptor coordinate mismatch. "
            "No angle calculation performed."
        )

    else:

        print(
            "\nStep 2: locate MET793 donor hydrogen"
        )

        donor = named_atom(
            hopt_atoms, "N"
        )["xyz"]

        hydrogen_candidates = [
            atom
            for atom in hopt_atoms
            if (
                atom["element"] == "H"
                and distance(
                    atom["xyz"], donor
                ) <= 1.25
            )
        ]

        if len(hydrogen_candidates) != 1:

            result["assessment"] = (
                "explicit_donor_hydrogen_unavailable"
            )

            print(
                "Expected one N-bound hydrogen; "
                f"found {len(hydrogen_candidates)}."
            )

        else:

            hydrogen = (
                hydrogen_candidates[0]["xyz"]
            )

            source = __import__(
                "pandas"
            ).read_csv(
                geo.SOURCE_CSV
            )

            rows = source.loc[
                source["phase4C_candidate_id"]
                == "EGFR_CAND_020"
            ]

            if len(rows) != 1:
                raise ValueError(
                    "CAND020 source lookup failed."
                )

            pose = Path(
                rows.iloc[0]["docked_pdbqt"]
            )

            ligand, _ = geo.read_pdbqt_top1(
                pose
            )

            # The previous trace identifies
            # heavy-atom index 15 as N [NA].
            acceptor_atom = ligand[14]

            if acceptor_atom["element"] != "N":
                raise ValueError(
                    "Expected ligand heavy atom "
                    "15 to be nitrogen."
                )

            acceptor = acceptor_atom["xyz"]

            d_a = distance(
                donor, acceptor
            )

            h_a = distance(
                hydrogen, acceptor
            )

            angle = angle_at_h(
                donor,
                hydrogen,
                acceptor,
            )

            plausible = (
                2.5 <= d_a <= 3.5
                and h_a <= 2.7
                and angle >= 135.0
            )

            result.update({
                "donor": "MET793 backbone N",
                "hydrogen_atom": (
                    hydrogen_candidates[0]["name"]
                ),
                "acceptor": (
                    "CAND020 PDBQT heavy atom 15 "
                    "N [NA]"
                ),
                "donor_acceptor_distance_A": (
                    round(d_a, 4)
                ),
                "hydrogen_acceptor_distance_A": (
                    round(h_a, 4)
                ),
                "D_H_A_angle_degrees": (
                    round(angle, 2)
                ),
                "passes_provisional_geometry": (
                    bool(plausible)
                ),
                "assessment": (
                    "provisional_geometry_plausible"
                    if plausible
                    else "geometry_not_supported_"
                    "by_provisional_thresholds"
                ),
            })

            print(
                "\nStep 3: N-H...N geometry"
            )

            print(
                f"Donor-acceptor distance: "
                f"{d_a:.4f} A"
            )

            print(
                f"H-acceptor distance: "
                f"{h_a:.4f} A"
            )

            print(
                f"N-H...N angle: "
                f"{angle:.2f} degrees"
            )

            print(
                "Provisional geometry plausible:",
                plausible
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nAssessment:", result["assessment"])
    print("Hydrogen bond confirmed: False")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
