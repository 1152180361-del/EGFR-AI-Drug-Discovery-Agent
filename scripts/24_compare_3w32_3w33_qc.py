from pathlib import Path
from collections import defaultdict
import re
import math


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

STRUCTURES = {
    "3W32": ROOT / "docking/receptor_screen/3W32/3W32.pdb",
    "3W33": ROOT / "docking/receptor_screen/3W33/3W33.pdb",
}


# ============================================================
# EGFR ATP-site regions used for receptor QC
# ============================================================

CRITICAL_REGIONS = {
    "P-loop": range(718, 727),
    "alphaC / ATP-site": range(742, 767),
    "hinge / Cys797 region": range(788, 801),
    "catalytic / DFG region": range(835, 859),
}

IMPORTANT_RESIDUES = [
    718, 719, 720, 721, 722, 723, 724, 725, 726,
    743, 745,
    762, 766,
    790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800,
    835, 855,
]


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}


WATER_NAMES = {
    "HOH", "WAT", "H2O", "DOD"
}


COMMON_SMALL_HET = {
    "SO4", "PO4", "GOL", "EDO", "PEG",
    "ACT", "FMT", "CL", "NA", "K",
    "CA", "MG", "ZN", "MN", "NI",
    "MES", "TRS", "BME", "DMS",
}


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def parse_pdb(path):

    atoms = []
    residues = set()
    protein_atoms = defaultdict(list)
    het_groups = defaultdict(list)

    missing_residues = []
    missing_atoms = []

    mutation_lines = []
    compnd_mutation_lines = []

    with open(path) as f:

        for line in f:

            # ------------------------------------------------
            # Mutation metadata
            # ------------------------------------------------

            if line.startswith("SEQADV"):
                if "MUTATION" in line.upper():
                    mutation_lines.append(
                        line.rstrip()
                    )

            if line.startswith("COMPND"):
                if "MUTATION" in line.upper():
                    compnd_mutation_lines.append(
                        line.rstrip()
                    )

            # ------------------------------------------------
            # Missing residues
            # REMARK 465
            # ------------------------------------------------

            if line.startswith("REMARK 465"):

                text = line[10:].strip()

                # Typical pattern:
                # GLY A 696
                match = re.search(
                    r"\b([A-Z0-9]{3})\s+([A-Za-z0-9])\s+(-?\d+)\b",
                    text
                )

                if match:

                    resname = match.group(1)
                    chain = match.group(2)
                    resnum = int(match.group(3))

                    if resname in STANDARD_AA:

                        entry = (
                            chain,
                            resnum,
                            resname
                        )

                        if entry not in missing_residues:
                            missing_residues.append(entry)

            # ------------------------------------------------
            # Missing atoms
            # REMARK 470
            # ------------------------------------------------

            if line.startswith("REMARK 470"):

                text = line[10:].strip()

                match = re.search(
                    r"\b([A-Z0-9]{3})\s+([A-Za-z0-9])\s+(-?\d+)\s+(.+)",
                    text
                )

                if match:

                    resname = match.group(1)
                    chain = match.group(2)
                    resnum = int(match.group(3))

                    if resname in STANDARD_AA:

                        atom_names = (
                            match.group(4)
                            .strip()
                            .split()
                        )

                        entry = {
                            "chain": chain,
                            "resnum": resnum,
                            "resname": resname,
                            "atoms": atom_names,
                        }

                        missing_atoms.append(
                            entry
                        )

            # ------------------------------------------------
            # Coordinate atoms
            # ------------------------------------------------

            if not line.startswith(
                ("ATOM", "HETATM")
            ):
                continue

            record = line[0:6].strip()
            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()

            try:
                resnum = int(
                    line[22:26]
                )

                x = float(
                    line[30:38]
                )

                y = float(
                    line[38:46]
                )

                z = float(
                    line[46:54]
                )

            except ValueError:
                continue

            element = (
                line[76:78]
                .strip()
                .upper()
            )

            if not element:

                element = re.sub(
                    r"[^A-Za-z]",
                    "",
                    atom_name
                )[:1].upper()

            atom = {
                "record": record,
                "atom": atom_name,
                "altloc": altloc,
                "resname": resname,
                "chain": chain,
                "resnum": resnum,
                "xyz": (x, y, z),
                "element": element,
            }

            atoms.append(atom)

            if (
                record == "ATOM"
                and resname in STANDARD_AA
            ):

                residue_key = (
                    chain,
                    resnum,
                    resname
                )

                residues.add(
                    residue_key
                )

                protein_atoms[
                    residue_key
                ].append(atom)

            elif record == "HETATM":

                key = (
                    chain,
                    resnum,
                    resname
                )

                het_groups[
                    key
                ].append(atom)

    return {
        "atoms": atoms,
        "residues": residues,
        "protein_atoms": protein_atoms,
        "het_groups": het_groups,
        "missing_residues": missing_residues,
        "missing_atoms": missing_atoms,
        "mutation_lines": mutation_lines,
        "compnd_mutation_lines": compnd_mutation_lines,
    }


def select_main_ligand(
    het_groups
):

    candidates = []

    for key, atoms in het_groups.items():

        chain, resnum, resname = key

        if resname in WATER_NAMES:
            continue

        heavy_atoms = [
            a
            for a in atoms
            if a["element"] != "H"
        ]

        n_heavy = len(
            heavy_atoms
        )

        # Large organic hetero groups are
        # much more likely to be the
        # co-crystal ligand than salts/buffer.
        if (
            n_heavy >= 10
            and resname not in COMMON_SMALL_HET
        ):

            candidates.append(
                (
                    n_heavy,
                    key,
                    heavy_atoms
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0]


def get_ligand_contacts(
    ligand_atoms,
    protein_atoms,
    cutoff=5.0,
):

    contacts = []

    for residue_key, atoms in protein_atoms.items():

        minimum_distance = None

        for pa in atoms:

            if pa["element"] == "H":
                continue

            for la in ligand_atoms:

                if la["element"] == "H":
                    continue

                d = distance(
                    pa["xyz"],
                    la["xyz"]
                )

                if (
                    minimum_distance is None
                    or d < minimum_distance
                ):

                    minimum_distance = d

        if (
            minimum_distance is not None
            and minimum_distance <= cutoff
        ):

            contacts.append(
                (
                    minimum_distance,
                    residue_key
                )
            )

    contacts.sort(
        key=lambda x: x[0]
    )

    return contacts


def inspect_structure(
    pdb_id,
    path
):

    data = parse_pdb(path)

    residues = data[
        "residues"
    ]

    protein_atoms = data[
        "protein_atoms"
    ]

    missing_residues = data[
        "missing_residues"
    ]

    missing_atoms = data[
        "missing_atoms"
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"{pdb_id} — EGFR RECEPTOR QC"
    )

    print(
        "=" * 72
    )

    print(
        "\nPDB file:",
        path
    )

    print(
        "Observed protein residues:",
        len(residues)
    )

    protein_atom_count = sum(
        len(v)
        for v
        in protein_atoms.values()
    )

    print(
        "Protein coordinate atoms:",
        protein_atom_count
    )

    # ========================================================
    # Mutation metadata
    # ========================================================

    print(
        "\n--- MUTATION METADATA ---"
    )

    if data["mutation_lines"]:

        for x in data[
            "mutation_lines"
        ]:

            print(x)

    elif data[
        "compnd_mutation_lines"
    ]:

        for x in data[
            "compnd_mutation_lines"
        ]:

            print(x)

    else:

        print(
            "No explicit MUTATION record found "
            "in PDB text."
        )

    # ========================================================
    # Missing residues
    # ========================================================

    print(
        "\n--- MISSING RESIDUES ---"
    )

    print(
        "Total missing residues:",
        len(missing_residues)
    )

    if missing_residues:

        for x in missing_residues:
            print(x)

    # ========================================================
    # Missing atoms
    # ========================================================

    print(
        "\n--- MISSING ATOMS ---"
    )

    print(
        "Residues with missing atoms:",
        len(missing_atoms)
    )

    if missing_atoms:

        for x in missing_atoms:
            print(
                x["chain"],
                x["resnum"],
                x["resname"],
                x["atoms"]
            )

    # ========================================================
    # Critical regions
    # ========================================================

    print(
        "\n--- ATP-SITE CRITICAL REGIONS ---"
    )

    critical_missing = []

    for region_name, region in (
        CRITICAL_REGIONS.items()
    ):

        region_missing = [
            x
            for x
            in missing_residues
            if x[1] in region
        ]

        print(
            f"{region_name}: "
            f"{len(region_missing)} "
            f"missing residues"
        )

        for x in region_missing:

            print(
                "   MISSING:",
                x
            )

            critical_missing.append(
                x
            )

    # ========================================================
    # Critical missing atoms
    # ========================================================

    critical_missing_atoms = []

    for item in missing_atoms:

        residue_number = (
            item["resnum"]
        )

        for region in (
            CRITICAL_REGIONS.values()
        ):

            if residue_number in region:

                critical_missing_atoms.append(
                    item
                )

                break

    print(
        "\nCritical-region residues "
        "with missing atoms:",
        len(critical_missing_atoms)
    )

    for x in critical_missing_atoms:

        print(
            "   ",
            x["chain"],
            x["resnum"],
            x["resname"],
            x["atoms"]
        )

    # ========================================================
    # Important residues
    # ========================================================

    print(
        "\n--- IMPORTANT RESIDUE PRESENCE ---"
    )

    observed_numbers = {
        resnum
        for chain, resnum, resname
        in residues
        if chain == "A"
    }

    missing_important = []

    for resnum in IMPORTANT_RESIDUES:

        present = (
            resnum
            in observed_numbers
        )

        print(
            f"A:{resnum:<4d}",
            "PASS"
            if present
            else "MISSING"
        )

        if not present:

            missing_important.append(
                resnum
            )

    # ========================================================
    # Cys797
    # ========================================================

    print(
        "\n--- CYS797 ---"
    )

    cys797_candidates = [
        key
        for key
        in protein_atoms
        if (
            key[0] == "A"
            and key[1] == 797
        )
    ]

    if cys797_candidates:

        key = (
            cys797_candidates[0]
        )

        atom_names = sorted(
            a["atom"]
            for a
            in protein_atoms[key]
        )

        print(
            "Residue:",
            key
        )

        print(
            "Atoms:",
            atom_names
        )

    else:

        print(
            "A:797 NOT FOUND"
        )

    # ========================================================
    # Hetero groups
    # ========================================================

    print(
        "\n--- NON-WATER HET GROUPS ---"
    )

    nonwater = []

    for key, atoms in (
        data["het_groups"].items()
    ):

        if key[2] in WATER_NAMES:
            continue

        heavy = [
            a
            for a in atoms
            if a["element"] != "H"
        ]

        nonwater.append(
            (
                key,
                len(heavy)
            )
        )

    nonwater.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for key, n in nonwater:

        print(
            key,
            "heavy_atoms =",
            n
        )

    # ========================================================
    # Main ligand detection
    # ========================================================

    ligand_info = (
        select_main_ligand(
            data["het_groups"]
        )
    )

    ligand_name = None
    ligand_heavy_count = None
    contacts = []

    if ligand_info is None:

        print(
            "\nNo suitable co-crystal "
            "ligand automatically detected."
        )

    else:

        (
            ligand_heavy_count,
            ligand_key,
            ligand_atoms
        ) = ligand_info

        ligand_name = (
            ligand_key[2]
        )

        print(
            "\n--- MAIN CO-CRYSTAL LIGAND ---"
        )

        print(
            "Ligand:",
            ligand_key
        )

        print(
            "Heavy atoms:",
            ligand_heavy_count
        )

        xs = [
            x["xyz"][0]
            for x
            in ligand_atoms
        ]

        ys = [
            x["xyz"][1]
            for x
            in ligand_atoms
        ]

        zs = [
            x["xyz"][2]
            for x
            in ligand_atoms
        ]

        center = (
            sum(xs) / len(xs),
            sum(ys) / len(ys),
            sum(zs) / len(zs),
        )

        print(
            "Ligand center:",
            tuple(
                round(v, 3)
                for v in center
            )
        )

        contacts = (
            get_ligand_contacts(
                ligand_atoms,
                protein_atoms,
                cutoff=5.0,
            )
        )

        print(
            "\n--- LIGAND 5 A CONTACT RESIDUES ---"
        )

        print(
            "Contact residues:",
            len(contacts)
        )

        for d, key in contacts:

            print(
                f"{key[0]}:{key[1]} "
                f"{key[2]:3s} "
                f"{d:5.2f} A"
            )

    # ========================================================
    # Does any reported missing atom occur in
    # an observed ligand-contact residue?
    # ========================================================

    contact_numbers = {
        key[1]
        for d, key
        in contacts
        if key[0] == "A"
    }

    contact_missing_atoms = [
        x
        for x
        in missing_atoms
        if x["resnum"]
        in contact_numbers
    ]

    print(
        "\nMissing-atom residues "
        "inside observed 5 A pocket:",
        len(contact_missing_atoms)
    )

    for x in contact_missing_atoms:

        print(
            "   ",
            x["chain"],
            x["resnum"],
            x["resname"],
            x["atoms"]
        )

    # ========================================================
    # Final structural QC status
    # ========================================================

    problems = []

    if critical_missing:

        problems.append(
            "missing critical ATP-site residues"
        )

    if critical_missing_atoms:

        problems.append(
            "missing atoms in critical ATP-site residues"
        )

    if missing_important:

        problems.append(
            "important ATP-site residue absent"
        )

    if ligand_info is None:

        problems.append(
            "co-crystal ligand not detected"
        )

    if contact_missing_atoms:

        problems.append(
            "missing atoms in ligand-contact residues"
        )

    print(
        "\n"
        + "-" * 72
    )

    if not problems:

        status = "PASS"

        print(
            f"{pdb_id} STRUCTURAL QC: PASS"
        )

    else:

        status = "REVIEW_REQUIRED"

        print(
            f"{pdb_id} STRUCTURAL QC: "
            f"REVIEW_REQUIRED"
        )

        for problem in problems:

            print(
                " -",
                problem
            )

    print(
        "-" * 72
    )

    return {
        "pdb_id": pdb_id,
        "status": status,
        "observed_residues": len(
            residues
        ),
        "missing_residues": len(
            missing_residues
        ),
        "critical_missing_residues": len(
            critical_missing
        ),
        "missing_atom_residues": len(
            missing_atoms
        ),
        "critical_missing_atom_residues": len(
            critical_missing_atoms
        ),
        "ligand": ligand_name,
        "ligand_heavy_atoms": ligand_heavy_count,
        "contacts_5A": len(
            contacts
        ),
        "contact_missing_atoms": len(
            contact_missing_atoms
        ),
    }


# ============================================================
# Run both structures
# ============================================================

summaries = []

for pdb_id, path in (
    STRUCTURES.items()
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    summaries.append(
        inspect_structure(
            pdb_id,
            path
        )
    )


# ============================================================
# Final comparison
# ============================================================

print(
    "\n\n"
    + "=" * 72
)

print(
    "3W32 vs 3W33 SUMMARY"
)

print(
    "=" * 72
)


header = (
    f"{'PDB':<8}"
    f"{'STATUS':<18}"
    f"{'MISS_RES':>10}"
    f"{'CRIT_RES':>10}"
    f"{'MISS_AT':>10}"
    f"{'CRIT_AT':>10}"
    f"{'LIGAND':>10}"
    f"{'5A_CONT':>10}"
)

print(header)

print(
    "-" * len(header)
)

for x in summaries:

    print(
        f"{x['pdb_id']:<8}"
        f"{x['status']:<18}"
        f"{x['missing_residues']:>10}"
        f"{x['critical_missing_residues']:>10}"
        f"{x['missing_atom_residues']:>10}"
        f"{x['critical_missing_atom_residues']:>10}"
        f"{str(x['ligand']):>10}"
        f"{x['contacts_5A']:>10}"
    )


print(
    "\nInterpretation:"
)

print(
    "Prefer a structure with:"
)

print(
    "1) zero missing residues in ATP-site regions"
)

print(
    "2) zero missing atoms in ligand-contact residues"
)

print(
    "3) intact Cys797 / hinge / P-loop"
)

print(
    "4) clearly resolved co-crystal ligand"
)

print(
    "5) successful self-redocking still required "
    "before production screening"
)
