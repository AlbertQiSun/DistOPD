"""The role list, in ONE place, with no heavy imports.

WHY THIS FILE EXISTS. `role_id` is stored in every packed cell as an INDEX INTO THIS LIST,
so two copies that drift do not raise anything -- they silently relabel every token in every
analysis that reads the cell. Three copies existed (harvest_token_table, role_effective_n,
fulldist_probe) and a test now asserts they are one.

It also has no torch import, which is what a CPU-only reader needs: probe_readback is pure
numpy and was pulling torch in transitively just to reach these eleven strings, so it could
not run in an environment without it.

THE ORDER IS FROZEN. Existing cells encode positions in this list; appending is safe,
reordering or removing is not.
"""
from __future__ import annotations

ROLE_VOCAB = ["other", "think_open", "think_close", "think_body", "search_open",
              "search_close", "query_body", "answer_open", "answer_close", "answer_body",
              "im_end"]

# The tokens that ARE the continue-vs-stop decision, as opposed to what gets said.
DECISION_ROLES = ("answer_close", "answer_open", "im_end", "search_open")
