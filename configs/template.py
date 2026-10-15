"""Frozen template + protocol constants — the SINGLE SOURCE OF TRUTH.

teacher/student x online/offline x SFT/OPD must all import token boundaries,
tags, and flags from HERE and nowhere else. Any drift in these values shifts
token boundaries and silently poisons every logprob alignment in the project
(see confound #1). Do not redefine any of these elsewhere.

Key decision (locked): enable_thinking = True. The teacher (MiroThinker, a
Qwen3-*-Thinking-lineage agentic model) carries its multi-hop search competence
in its NATIVE <think> channel. Under enable_thinking=False Qwen3's template
pre-fills a CLOSED empty block `<think>\n\n</think>` right after the assistant
generation prompt; the model reads "thinking is already done" and jumps straight
to <search>/<answer>, producing reasoning-free, repetitive, non-terminating
rollouts (verified: scripts/probe_think_template.py -- the False tail carries the
empty block, the True tail does not). enable_thinking=True emits no such block,
so the model reasons natively.

The classic reason to AVOID enable_thinking=True -- Qwen3's multi-turn template
strips previous-turn thinking on RE-RENDER, so a turn's prefix would tokenize
DIFFERENTLY from what was generated (confound #1) -- does NOT apply here: this
project NEVER re-renders. apply_chat_template runs ONCE for the initial prompt;
every later segment (including the model's native <think> tokens) is appended at
the token-id level and never round-tripped. The native <think> tokens simply
become ordinary ACTION tokens in the frozen flat sequence -- exactly the "reason
with explicit <think> tags that are ordinary action tokens" design intent, which
enable_thinking=False was silently defeating.

Bonus: enable_thinking=True's assistant-prompt tail is `<|im_start|>assistant\n`
-- byte-identical to the mock render_prompt() literal -- eliminating the real-vs-
mock divergence that hid the empty-think bug through the entire M2-M5 mock suite.

Flipping this flag DELIBERATELY changes template_fingerprint (Gate 1), which
invalidates any SFT data / pi_ref produced under the old flag: they MUST be
regenerated / retrained, and the provenance gate must keep old- and new-flag
shards from mixing.
"""
from __future__ import annotations

import os

# --- Qwen3 chat-template flags (frozen for the whole project) ---
ENABLE_THINKING = True         # native thinking channel ON: teacher is Thinking-lineage;
                               # we never re-render, so confound #1's re-render strip never fires
ADD_GENERATION_PROMPT = True   # applied ONLY to the one-shot initial prompt render

# --- protocol tags (explicit text tags, Search-R1 style) ---
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
SEARCH_OPEN = "<search>"
SEARCH_CLOSE = "</search>"
INFO_OPEN = "<information>"
INFO_CLOSE = "</information>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
# environment control-signal tag. Emitted by the rollout ENV (never the model) on the
# forced-final-answer turn; wrapped as an OBSERVATION span (masked) like <information>.
WARNING_OPEN = "<warning>"
WARNING_CLOSE = "</warning>"

# environment-injected observation wrapper. The ENTIRE block (prefix + tags +
# docs + suffix) is OBSERVATION => masked out of loss/advantage, but STILL part
# of the sequence so later action tokens are conditioned on it.
OBS_PREFIX = "\n\n"
OBS_SUFFIX = "\n\n"

# chat scaffold tokens (Qwen3 ChatML)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# NOTE: the answer-format example is a PLACEHOLDER (<answer>...</answer>), NOT a concrete named
# entity. An earlier version used `<answer>Beijing</answer>`; the 64-Q HotpotQA smoke showed
# 12.5% of FORCED answers echoing the literal "Beijing" when the model hit the search budget
# with no evidence -- a concrete example answer becomes an attractor the model copies under
# uncertainty, poisoning ~8% of the shard with a fixed wrong token. The placeholder removes that
# attractor and unifies with FORCED_ANSWER_WARNING / DUPLICATE_QUERY_HINT (both already use
# <answer>...</answer>). Editing SYSTEM_PROMPT DELIBERATELY changes template_fingerprint (Gate 1),
# invalidating any SFT data / pi_ref made under the Beijing prompt -- regenerate + retrain (which
# we were about to do anyway = no extra cost) and keep old/new shards from mixing.
#
# NOTE (anti-over-confirmation, A/B pending): the sft_shard1 diagnostic showed 69% of forced NQ
# 1-hop trajectories had ALREADY surfaced the gold answer (it was literally typed into later
# queries: "...1993", "...Jim Gilstrap", "...July 28 1914") yet kept searching to CONFIRM instead
# of emitting <answer> -- MiroThinker's dominant non-termination mode is over-confirmation, NOT
# dup-spinning (exact-dup only 0.56 turns/traj). The prior four instructions target "don't know
# when to stop" / "don't repeat" but say NOTHING about "I already have the answer but want to
# re-check it once more"; the sentence below is a SINGLE-VARIABLE addition aimed precisely at that
# mode. This bumps template_fingerprint again -- provisional pending the 64-Q HotpotQA + NQ-1hop
# A/B (control = sft_shard1 per-source natural: 1hop 25.21%, 2hop 25.42%). If natural_frac lifts
# materially, keep it and full-regen; if it doesn't move, REVERT this commit so sft_shard1 stays
# provenance-valid under fingerprint ceb7aae9b67cd471 for the route-1 (accept-as-is) path.
_V2_TERSE = (
    "You are a research assistant that answers questions by searching a corpus. "
    "Reason inside <think>...</think>. To search, emit <search>your query</search> "
    "and stop; the system returns <information>...</information>. Your search budget "
    "is limited, so search only when you need a new fact and make every query count. "
    "Do not repeat a query you have already searched -- it returns the same documents. "
    "As soon as you have enough to answer, stop searching and give your answer as "
    "<answer>...</answer>. The moment the answer to the question appears in the "
    "retrieved documents, answer immediately -- do not run another search just to "
    "re-confirm or double-check a fact you already have. If the searches do not "
    "fully answer the question, give your single best answer anyway rather than "
    "searching in circles."
)

# --- v1_asp: the prompt this project's predecessor (ASP) used, ported ---------------------
#
# Ported, not rewritten. Two changes only, both forced by THIS environment: the original said
# the model would observe "a piece of summarized search results" (ASP ran a summarizer; this
# pipeline returns raw retrieved passages), and the trailing "### Question {question}" block
# is dropped because rollout.render_prompt already puts the question in the USER turn.
# Everything else -- the worked example, all seven reminders, the query-format section -- is
# verbatim, INCLUDING the parts that carry known risk:
#
#   * the example's <information> blocks name concrete entities (French / MacMahon / Sint
#     Maarten). A concrete example ANSWER was removed from _V2_TERSE for exactly this reason:
#     the 64-Q smoke found 12.5% of forced answers echoing the literal "Beijing". The example
#     here keeps a PLACEHOLDER in the answer slot, but its evidence text does not. Neutering
#     it would make the A/B test something other than the prompt under test, so it is kept
#     verbatim and the echo is MEASURED instead (scripts/compare_prompt_ab.py counts it).
#   * reminder 2 ("Do not use your own knowledge ... to speculate") is in tension with this
#     env's FORCED_ANSWER_WARNING ("using the information above and what you already know").
#     Kept as-is; if v1 wins, that seam is the first thing to re-read.
#   * reminder 7 pins the year to 2018. That happens to be RIGHT for this corpus (DPR
#     wiki-18, dump 2018-12-20), but it is a bundled variable, not a free one.
#
# This is a BUNDLED change: v1 ADDS a worked example, short-answer instructions, a trust-the-
# retrieval instruction and query-format guidance, while REMOVING v2's search-budget,
# no-duplicate and anti-re-confirmation sentences. A win says "switch"; it does NOT say which
# sentence did the work. _V3_TERSE_SHORTANS below is the single-variable follow-up.
_V1_ASP = (
    "Answer the following question from the user with the help of a Wikipedia search "
    "engine. Please reason step by step. You should think about what you need to know "
    "in order to answer the question, and then search for that information using the "
    "search engine. To perform a search operation, write a web search question and "
    "enclose it with <search> and </search>. You will immediately observe the retrieved "
    "passages within the <information> and </information> tags. You can then use this "
    "retrieved information to continue your reasoning. You can repeat the search process "
    "many times. Once you think you have all the information you need, you can end the "
    "thinking process and provide the final answer. You MUST enclose your final answer "
    "with <answer> and </answer>.\n"
    "\n"
    "### Example\n"
    "Question: When did the people who captured Malakoff come to the region where "
    "Philipsburg is located?\n"
    "\n"
    "Your thinking process:\n"
    "Alright, I need to figure out when the people who captured Malakoff came to the "
    "region where Philipsburg is located. I should first find out who were the people "
    "that captured Malakoff. Let me write a search question to look it up with the "
    "Wikipedia search engine.\n"
    "\n"
    "<search> Who were the people that captured Malakoff? </search>\n"
    "<information> The French army under General MacMahon successfully captured the "
    "Malakoff redoubt on 8th. </information>\n"
    "\n"
    "Okay, so the French people captured Malakoff. Now, the next step would be to figure "
    "out in what region Philipsburg is located. I will write a web search to look that "
    "up.\n"
    "\n"
    "<search> Where is Philipsburg located at? </search>\n"
    "<information> Philipsburg is the main town and capital of Sint Maarten, a "
    "constituent country of the Kingdom of the Netherlands. </information>\n"
    "\n"
    "...[more thoughts shortened]...\n"
    "\n"
    "Your final response:\n"
    "<answer> [your concise short-form final answer goes here] </answer>\n"
    "\n"
    "### Reminders\n"
    "1. You should carefully follow the format of searching and answering as shown in "
    "the example above.\n"
    "2. You should always and directly use the Wikipedia search engine to look up the "
    "information needed to answer the question. Do not use your own knowledge or "
    "personal experiences to speculate.\n"
    "3. Your search queries should be a complete, natural language question instead of "
    "keywords. For instance, instead of searching for \"people that captured Malakoff\", "
    "you should search for \"Who were the people that captured Malakoff?\".\n"
    "4. You can trust the information retrieved from the search engine to be accurate "
    "and factual, and use it in your subsequent reasoning. No need to reflect and doubt "
    "it.\n"
    "5. Your final answer should be a short-form answer: give only the answer itself (a "
    "name, date, number, or short phrase), not a full sentence. Do NOT restate the "
    "question or add explanations.\n"
    "6. Your final response should only have the answer enclosed in <answer> and "
    "</answer> tags. Do not include any other information or text.\n"
    "7. Assume that the current year is 2018. No need to look up information that is "
    "more recent than this year.\n"
    "\n"
    "### Search Query Format Guidelines\n"
    "When writing search queries, follow these specific formats depending on what "
    "information you need:\n"
    "\n"
    "**Format A: When inquiring about an attribute of an entity**\n"
    "Use the pattern: \"wh-word is [attribute] of [entity]\"\n"
    "Examples:\n"
    "- \"What is the capital of France?\" (asking about the capital attribute of France "
    "entity)\n"
    "- \"When is the birthday of Albert Einstein?\" (asking about the birthday attribute "
    "of Albert Einstein entity)\n"
    "- \"Where is the location of Mount Everest?\" (asking about the location attribute "
    "of Mount Everest entity)\n"
    "\n"
    "**Format B: When inquiring about entities that satisfy given values on given "
    "attributes**\n"
    "Use the pattern: \"wh-word [satisfies the values on given attributes]\"\n"
    "Examples:\n"
    "- \"Who was born in 1879?\" (asking about entities with birth year attribute = "
    "1879)\n"
    "- \"Which countries have a population over 100 million?\" (asking about entities "
    "with population attribute > 100 million)\n"
    "- \"What cities are located in California?\" (asking about entities with location "
    "attribute = California)\n"
    "\n"
    "Choose the appropriate format based on whether you're looking for an attribute "
    "value of a specific entity (Format A) or searching for entities that match certain "
    "criteria (Format B).\n"
    "\n"
    "Now let's begin!"
)

# --- v3: _V2_TERSE plus ONE thing -- what the answer span may CONTAIN ----------------------
#
# The single-variable arm. _V2_TERSE says where to put the answer and never says what may be
# in it: no length instruction, no "only the answer", no "do not restate the question". The
# observed failures (90-word answer spans, deliberation text inside the span, an <answer> the
# model opens and never closes) are all consistent with a model that was never told. This
# adds ASP's reminders 5 and 6 and NOTHING else, so if it recovers most of a v1 win, the
# worked example / query-format section / trust instruction are not what mattered -- and the
# cheap change is the one to keep, since it costs ~50 prompt tokens instead of ~800 and
# carries no example-echo risk.
_V3_TERSE_SHORTANS = _V2_TERSE + (
    " Your answer must be short-form: give only the answer itself (a name, date, number, "
    "or short phrase), not a full sentence, and do not restate the question or add "
    "explanations. Everything between <answer> and </answer> must be the answer and "
    "nothing else."
)

PROMPT_VARIANTS = {
    "v2_terse": _V2_TERSE,                    # the DEFAULT -- fingerprint f9da237f476eb71e
    "v1_asp": _V1_ASP,
    "v3_terse_shortans": _V3_TERSE_SHORTANS,
}

# OPD_PROMPT_VARIANT selects the arm. This is a NUMERICS-CHANGING env var by design, which is
# normally forbidden here (see CHANNEL_ALLOWLIST) -- it is allowed only because SYSTEM_PROMPT
# is already inside template_fingerprint(), so gate 1 catches cross-variant mixing on its own,
# and because sft/datagen writes the variant NAME and the full prompt TEXT into stats.json (a
# hash tells you two runs differ; it does not tell you how). The default must stay
# byte-identical to the prompt every existing artifact was produced under --
# tests/test_prompt_variants.py pins that fingerprint literally.
# DEFAULT CHANGED 2026-08-22, v2_terse -> v3_terse_shortans, on the ab4 round -- the first
# round where the action cap, the prompt variant and the code version were all correct at
# once. Across all three teachers v3 never lowered natural termination (MiroThinker +0.0,
# Qwen3-32B +1.6, Qwen3-30B-A3B +10.9) and took the whole answer-side gain: median answer
# 53 -> 2 words on MiroThinker with in-span deliberation 44.1% -> 3.2%, 30 -> 2 and
# 13.3% -> 0.0% on Qwen3-32B, and MiroThinker's not_terminated 7.8% -> 0.0%.
#
# NOT claimed: that v3 makes models terminate on their own. The +12.5 / +14.1 natural gains
# seen in earlier rounds did NOT replicate once max_action_tokens stopped clipping actions --
# they were truncation artifacts. What v3 changes is whether the model COMPLETES an answer
# span the environment has already opened, which is exactly the quantity Row 10 found OPD
# damaging, and it moves that on all three teachers.
#
# This invalidates sft_shard1, pi_ref and the Row 10 precompute -- but max_action_tokens
# 512 -> 2560 already did, so the regeneration was owed either way.
PROMPT_VARIANT = os.environ.get("OPD_PROMPT_VARIANT", "v3_terse_shortans")
if PROMPT_VARIANT not in PROMPT_VARIANTS:
    raise ValueError(
        f"OPD_PROMPT_VARIANT={PROMPT_VARIANT!r} is not a registered prompt variant; "
        f"choose one of {sorted(PROMPT_VARIANTS)}")
SYSTEM_PROMPT = PROMPT_VARIANTS[PROMPT_VARIANT]

# --- environment-injected control observations (both OBSERVATION => masked) ---
# These are appended by the rollout environment, NOT generated by the model, so they are
# tagged OBSERVATION exactly like <information>: masked out of loss/advantage but still
# conditioned on by the subsequent SUPERVISED action. Do NOT tag these ACTION.

# FORCED_ANSWER_WARNING: injected when the search budget is exhausted (rollout Guard 1).
# It converts a would-be non-terminating trajectory into one final SUPERVISED answer turn
# -- the environment seam (Search-R1/StepSearch/s3 set do_search=False on the last turn)
# this pipeline was missing. The forced answer that follows is an ACTION span (supervised).
FORCED_ANSWER_WARNING = (
    f"{WARNING_OPEN}Your search budget is now exhausted. You cannot search again. "
    f"Answer the question now using the information above and what you already know. "
    f"Give your single best answer even if you are not certain, as "
    f"{ANSWER_OPEN}...{ANSWER_CLOSE}.{WARNING_CLOSE}"
)

# DUPLICATE_QUERY_HINT: injected when the model repeats a query it already ran (rollout
# Guard 2). Retrieval is deterministic, so the repeat would return identical documents and
# zero new evidence; this nudges a DIFFERENT query or an answer instead.
DUPLICATE_QUERY_HINT = (
    f"{INFO_OPEN}You have already searched this exact query; it returns the same "
    f"documents shown above and no new information. Try a different query, or answer "
    f"now with {ANSWER_OPEN}...{ANSWER_CLOSE}.{INFO_CLOSE}"
)
