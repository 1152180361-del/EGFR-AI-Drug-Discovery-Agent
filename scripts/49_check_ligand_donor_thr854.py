from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd

audit = import_module(
    "scripts.48_compare_egfr_polar_contacts"
)
geo = audit.geo

SOURCE = geo.SOURCE_CSV
OUTPUT = (
    geo.OUTPUT
    / "Ligand_Donor_THR854_Check.csv"
)

TARGETS = {
    "EGFR_CAND_003": 3,
    "EGFR_CAND_020": 19,
}


def read_explicit_hydrogens(path):
    atoms = []

    for line in path.read_text(
        errors="replace"
    ).splitlines():

        if line.startswith("ENDMDL"):
            break

        if line[:6].strip() not in (
            "ATOM", "HETATM"
        ):
            continue

        typ = line.split()[-1].upper()

        if typ not in ("HD", "H", "HS"):
            continue

        atoms.append({
            "name": line[12:16].strip(),
            "type": typ,
            "xyz": geo.xyz_from_line(line)
        })

    return atoms


def distance(a, b):
    return float(np.linalg.norm(a - b))


def main():
    source = pd.read_csv(
        SOURCE
    ).set_index(
        "phase4C_candidate_id"
    )

    protein = geo.read_pdb(
        geo.RECEPTOR, {"ATOM"}
    )

    threonine = [
        atom for atom in protein
        if atom["chain"] == "A"
        and atom["resi"] == "854"
        and atom["resname"] == "THR"
        and atom["name"] == "OG1"
    ]

    if len(threonine) != 1:
        raise RuntimeError(
            "THR854 OG1 not uniquely identified."
        )

    acceptor = threonine[0]["xyz"]
    results = []

    for cid, number in TARGETS.items():
        ligand, mapping, acc, donors, err = (
            audit.chemistry(cid, source)
        )

        donor_atom = ligand[number - 1]
        sdf_index = mapping[number - 1]

        if sdf_index not in donors:
            raise RuntimeError(
                f"{cid}: target N is not "
                "classified as a donor."
            )

        pose_file = Path(
            source.loc[cid, "docked_pdbqt"]
        )

        hydrogens = read_explicit_hydrogens(
            pose_file
        )

        donor_xyz = donor_atom["xyz"]

        attached = [
            h for h in hydrogens
            if 0.70 <= distance(
                h["xyz"], donor_xyz
            ) <= 1.25
        ]

        da = distance(
            donor_xyz, acceptor
        )

        print("\n", cid)
        print("Ligand donor atom:", number)
        print("SDF atom:", sdf_index + 1)
        print("N...OG1 distance:", round(da, 4))
        print(
            "Explicit attached H atoms:",
            len(attached)
        )

        if not attached:
            results.append({
                "candidate": cid,
                "ligand_donor_index": number,
                "donor_acceptor_A": da,
                "hydrogen_acceptor_A": None,
                "D_H_A_angle_deg": None,
                "status": "explicit_H_unavailable"
            })
            continue

        for h in attached:
            ha = distance(
                h["xyz"], acceptor
            )

            dha = audit.angle(
                donor_xyz,
                h["xyz"],
                acceptor
            )

            plausible = (
                2.5 <= da <= 3.5
                and ha <= 2.7
                and dha >= 135.0
            )

            status = (
                "provisional_geometry_pass"
                if plausible
                else "geometry_not_supported"
            )

            print("Hydrogen:", h["name"])
            print("H...OG1:", round(ha, 4))
            print("N-H...OG1:", round(dha, 2))
            print("Assessment:", status)

            results.append({
                "candidate": cid,
                "ligand_donor_index": number,
                "hydrogen_name": h["name"],
                "donor_acceptor_A": da,
                "hydrogen_acceptor_A": ha,
                "D_H_A_angle_deg": dha,
                "status": status,
            })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(results).to_csv(
        OUTPUT, index=False
    )

    print("\nSaved:", OUTPUT)
    print(
        "Geometry is provisional; "
        "no stable hydrogen bond is established."
    )


if __name__ == "__main__":
    main()
