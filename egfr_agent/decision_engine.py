import numpy as np
import pandas as pd


class EGFRDecisionEngine:

    REQUIRED = (
        "phase4C_candidate_id",
        "candidate_track",
        "consensus_rank",
        "qsar_predicted_pIC50",
        "key_ATP_contact_coverage",
        "MET793_distance",
    )

    VALID_TRACKS = {
        "SUPPORTED_A",
        "SUPPORTED_B_HIGH_SPREAD",
        "EXPLORATORY_OOD",
    }

    @classmethod
    def validate_input(cls, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Decision input must be a DataFrame")

        if df.empty:
            raise ValueError("Decision input is empty")

        missing = set(cls.REQUIRED) - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing decision fields: {sorted(missing)}"
            )

        frame = df.copy()

        ids = (
            frame["phase4C_candidate_id"]
            .astype("string")
            .str.strip()
        )

        if ids.isna().any() or ids.eq("").any():
            raise ValueError("Empty candidate ID")

        if ids.duplicated().any():
            raise ValueError("Duplicate candidate IDs")

        frame["phase4C_candidate_id"] = ids.astype(str)

        tracks = (
            frame["candidate_track"]
            .astype("string")
            .str.strip()
        )

        invalid = tracks.isna() | ~tracks.isin(
            cls.VALID_TRACKS
        )

        if invalid.any():
            raise ValueError(
                "Invalid candidate_track: "
                + str(tracks[invalid].tolist())
            )

        frame["candidate_track"] = tracks.astype(str)

        numeric_fields = (
            "consensus_rank",
            "qsar_predicted_pIC50",
            "key_ATP_contact_coverage",
            "MET793_distance",
        )

        for column in numeric_fields:
            original = frame[column]

            if original.map(
                lambda v: isinstance(v, (bool, np.bool_))
            ).any():
                raise ValueError(
                    f"Boolean value in {column}"
                )

            try:
                values = pd.to_numeric(
                    original, errors="raise"
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid numeric values in {column}"
                ) from exc

            numbers = values.to_numpy(dtype=float)

            if not np.isfinite(numbers).all():
                raise ValueError(
                    f"Missing or non-finite values in {column}"
                )

            frame[column] = numbers

        rank = frame["consensus_rank"].to_numpy()

        if (
            (rank < 1).any()
            or (rank != np.floor(rank)).any()
        ):
            raise ValueError(
                "consensus_rank must be a positive integer"
            )

        coverage = frame[
            "key_ATP_contact_coverage"
        ].to_numpy()

        if ((coverage < 0) | (coverage > 1)).any():
            raise ValueError(
                "ATP contact coverage must be within [0, 1]"
            )

        if (frame["MET793_distance"] <= 0).any():
            raise ValueError(
                "MET793_distance must be positive"
            )

        return frame

    def analyze(self, df):
        frame = self.validate_input(df)
        results = []

        for _, row in frame.iterrows():
            score = 0
            reasons = []

            track = row["candidate_track"]

            if track == "SUPPORTED_A":
                score += 3
                reasons.append("QSAR-supported candidate")

            elif track == "EXPLORATORY_OOD":
                score += 1
                reasons.append(
                    "Exploratory structural candidate"
                )

            rank = row["consensus_rank"]

            if rank == 1:
                score += 5
                reasons.append("Consensus rank #1")

            elif rank == 2:
                score += 4
                reasons.append("Consensus rank #2")

            elif rank <= 5:
                score += 3
                reasons.append(
                    f"Consensus rank #{int(rank)} "
                    "(top-5 rank contribution)"
                )

            pic50 = row["qsar_predicted_pIC50"]

            if pic50 >= 7:
                score += 2
                reasons.append(
                    "High QSAR predicted activity"
                )

            elif pic50 >= 6.8:
                score += 1

            atp = row["key_ATP_contact_coverage"]

            if atp >= 0.7:
                score += 2
                reasons.append(
                    "High docking-derived ATP-pocket "
                    "contact coverage"
                )

            met = row["MET793_distance"]

            if met <= 4:
                score += 1
                reasons.append(
                    "Short MET793 proximity "
                    "(distance descriptor; "
                    "H-bond unverified)"
                )

            results.append({
                "candidate": row["phase4C_candidate_id"],
                "score": score,
                "reason": "; ".join(reasons),
            })

        return (
            pd.DataFrame(results)
            .sort_values("score", ascending=False)
        )
