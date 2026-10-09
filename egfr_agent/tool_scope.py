import re


class EGFRToolScopeResolver:
    """
    Deterministic tool authorization.

    This is a conservative rule-based scope guard,
    not a general natural-language understanding model.
    """

    ALL_TOOLS = (
        "qsar_prediction",
        "structure_analysis",
    )

    @staticmethod
    def _matches(text, patterns):
        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def resolve(self, user_request):

        if not isinstance(user_request, str):
            raise TypeError(
                "user_request must be a string."
            )

        text = user_request.strip().lower()

        if not text:
            raise ValueError(
                "user_request cannot be empty."
            )

        # Activity / QSAR intent
        qsar_patterns = [
            r"\bqsar\b",
            r"\bpic50\b",
            r"\bic50\b",
            r"\bactivity\b",
            r"\bpotency\b",
            r"\bpredict(?:ion|ions|ed)?\b",
            r"\bzinc\b",
            r"活性",
            r"预测",
            r"效力",
        ]

        # Explicit structural evidence intent.
        # A generic word such as "ranked" is NOT
        # sufficient to authorize structure analysis.
        structure_patterns = [
            r"\bdocking\b",
            r"\bdock(?:ed)?\b",
            r"\bstructure\b",
            r"\bstructural\b",
            r"\bpose(?:s)?\b",
            r"\bhinge\b",
            r"\bmet793\b",
            r"\batp\b",
            r"\bpocket\b",
            r"结构",
            r"对接",
            r"构象",
            r"口袋",
            r"铰链",
        ]

        # Candidate-specific ranking/comparison needs
        # the existing candidate evidence table.
        candidate_comparison = (
            (
                "egfr_cand_" in text
                or "candidate" in text
                or "候选" in text
            )
            and self._matches(
                text,
                [
                    r"\bcompar(?:e|ison)\b",
                    r"\brank(?:ing|ed)?\b",
                    r"\bprioriti[sz]",
                    r"比较",
                    r"排名",
                    r"优先",
                    r"推荐",
                ],
            )
        )

        has_qsar = self._matches(
            text, qsar_patterns
        )

        has_structure = (
            self._matches(text, structure_patterns)
            or candidate_comparison
        )

        # Explicit exclusions.
        # Negation must take precedence over
        # positive keyword matches.
        negative_structure_patterns = [
            (
                r"\b(?:do\s+not|don't|dont|without|"
                r"avoid|skip|exclude|no\s+need\s+to)\b"
                r"[^.;\n]{0,70}"
                r"\b(?:structure|structural|docking|"
                r"dock|pose\s+analysis)\b"
            ),
            (
                r"\b(?:structure\s+analysis|"
                r"structural\s+analysis|docking)\b"
                r"[^.;\n]{0,30}"
                r"\b(?:not\s+needed|not\s+requested)\b"
            ),
            r"(?:不要|无需|不需要|不用|禁止|跳过)"
            r"[^。；，\n]{0,20}"
            r"(?:结构分析|结构工具|分子对接|对接分析)",
        ]

        negative_qsar_patterns = [
            (
                r"\b(?:do\s+not|don't|dont|without|"
                r"avoid|skip|exclude|no\s+need\s+to)\b"
                r"[^.;\n]{0,70}"
                r"\b(?:qsar|activity\s+prediction|"
                r"potency\s+prediction)\b"
            ),
            r"(?:不要|无需|不需要|不用|禁止|跳过)"
            r"[^。；，\n]{0,20}"
            r"(?:QSAR|活性预测|效力预测)",
        ]

        prohibit_structure = self._matches(
            text, negative_structure_patterns
        )

        prohibit_qsar = self._matches(
            text, negative_qsar_patterns
        )

        # Explicit single-domain requests.
        only_qsar = self._matches(
            text,
            [
                r"\b(?:only|just)\s+(?:the\s+)?"
                r"(?:global\s+)?(?:zinc\s+)?"
                r"(?:qsar|activity\s+predictions|pic50)\b",
                r"(?:只|仅)(?:看|读取|分析|使用)?"
                r"(?:QSAR|活性预测|pIC50)",
            ],
        )

        only_structure = self._matches(
            text,
            [
                r"\b(?:only|just)\s+(?:the\s+)?"
                r"(?:structure|structural|docking)\b",
                r"(?:只|仅)(?:看|读取|分析|使用)?"
                r"(?:结构|对接)",
            ],
        )

        if only_qsar:
            prohibit_structure = True

        if only_structure:
            prohibit_qsar = True

        allowed = []

        if has_qsar and not prohibit_qsar:
            allowed.append("qsar_prediction")

        if has_structure and not prohibit_structure:
            allowed.append("structure_analysis")

        if not allowed:
            raise ValueError(
                "No tool can be safely authorized "
                "from this request. Please specify "
                "whether you want existing QSAR "
                "predictions or structural evidence."
            )

        prohibited = [
            name
            for name in self.ALL_TOOLS
            if name not in allowed
        ]

        return {
            "allowed_tools": allowed,
            "prohibited_tools": prohibited,
            "explicit_structure_exclusion": (
                bool(prohibit_structure)
            ),
            "explicit_qsar_exclusion": (
                bool(prohibit_qsar)
            ),
            "authorization_method": (
                "deterministic_request_scope_guard"
            ),
        }
