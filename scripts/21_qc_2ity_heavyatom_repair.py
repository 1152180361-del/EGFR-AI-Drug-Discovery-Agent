from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]

BEFORE = (
    ROOT
    / "docking"
    / "2ITY"
    / "receptor"
    / "2ITY_clean_receptor.pdb"
)

AFTER = (
    ROOT
    / "docking"
    / "2ITY"
    / "receptor"
    / "2ITY_heavyfixed.pdb"
)


TARGETS = {
    860: "LYS",
    988: "HIS",
}


EXPECTED_ATOMS = {

    860: {
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
    },

    988: {
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
    },
}


def parse_pdb(path):

    residues = defaultdict(
        list
    )

    atoms_total = 0

    with open(
        path,
        "r"
    ) as handle:

        for line in handle:

            if not line.startswith(
                ("ATOM", "HETATM")
            ):
                continue

            atoms_total += 1

            chain = line[21].strip()

            try:
                resseq = int(
                    line[22:26]
                )
            except Exception:
                continue

            resname = (
                line[17:20]
                .strip()
            )

            atom_name = (
                line[12:16]
                .strip()
            )

            element = (
                line[76:78]
                .strip()
            )

            if (
                chain == "A"
                and element != "H"
            ):

                residues[
                    (
                        resseq,
                        resname
                    )
                ].append(
                    atom_name
                )

    return (
        residues,
        atoms_total
    )


before, before_atoms = (
    parse_pdb(
        BEFORE
    )
)

after, after_atoms = (
    parse_pdb(
        AFTER
    )
)


print(
    "\n================================="
)

print(
    "PHASE 3.4B — HEAVY ATOM REPAIR QC"
)

print(
    "================================="
)


print(
    "Atoms before:",
    before_atoms
)

print(
    "Atoms after:",
    after_atoms
)

print(
    "Atoms added:",
    after_atoms
    - before_atoms
)


# =========================================
# Residue count
# =========================================

before_resnums = {
    key[0]
    for key in before
}

after_resnums = {
    key[0]
    for key in after
}


print(
    "\nResidues before:",
    len(
        before_resnums
    )
)

print(
    "Residues after:",
    len(
        after_resnums
    )
)


new_residues = (
    after_resnums
    - before_resnums
)


print(
    "New residue numbers added:",
    sorted(
        new_residues
    )
)


# =========================================
# Target residues
# =========================================

overall_pass = True


for resseq, expected_name in (
    TARGETS.items()
):

    before_key = None
    after_key = None


    for key in before:

        if key[0] == resseq:

            before_key = key


    for key in after:

        if key[0] == resseq:

            after_key = key


    print(
        "\n---------------------------------"
    )

    print(
        f"A:{resseq} {expected_name}"
    )

    print(
        "---------------------------------"
    )


    if before_key is None:

        print(
            "ERROR: residue absent before repair"
        )

        overall_pass = False

        continue


    if after_key is None:

        print(
            "ERROR: residue absent after repair"
        )

        overall_pass = False

        continue


    before_atoms_set = set(
        before[
            before_key
        ]
    )

    after_atoms_set = set(
        after[
            after_key
        ]
    )


    print(
        "Before:"
    )

    print(
        sorted(
            before_atoms_set
        )
    )


    print(
        "After:"
    )

    print(
        sorted(
            after_atoms_set
        )
    )


    added_atoms = (
        after_atoms_set
        - before_atoms_set
    )


    print(
        "Added:"
    )

    print(
        sorted(
            added_atoms
        )
    )


    missing_after = (
        EXPECTED_ATOMS[
            resseq
        ]
        - after_atoms_set
    )


    unexpected_after = (
        after_atoms_set
        - EXPECTED_ATOMS[
            resseq
        ]
    )


    print(
        "Missing after repair:"
    )

    print(
        sorted(
            missing_after
        )
    )


    print(
        "Unexpected atoms:"
    )

    print(
        sorted(
            unexpected_after
        )
    )


    if missing_after:

        overall_pass = False


# =========================================
# Protect known unresolved regions
# =========================================

known_missing_blocks = [
    range(
        866,
        876
    ),
    range(
        990,
        1003
    ),
]


unexpected_rebuilt = []


for block in known_missing_blocks:

    for resseq in block:

        if (
            resseq
            not in before_resnums
            and resseq
            in after_resnums
        ):

            unexpected_rebuilt.append(
                resseq
            )


print(
    "\nUnexpected rebuilt missing residues:"
)

print(
    unexpected_rebuilt
)


if unexpected_rebuilt:

    overall_pass = False


# =========================================
# Final decision
# =========================================

print(
    "\n================================="
)

if overall_pass:

    print(
        "HEAVY-ATOM REPAIR STATUS: PASS"
    )

else:

    print(
        "HEAVY-ATOM REPAIR STATUS: FAIL"
    )

print(
    "================================="
)
