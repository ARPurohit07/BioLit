"""Task-specific prompt builders.

Each function returns (system_prompt, user_prompt). There is deliberately no
single mega-prompt: every generation task gets its own focused system prompt
so instructions stay unambiguous for a small local model.

All prompts share the same evidence formatting and the same citation
discipline: cite only from the numbered evidence blocks using [n] markers,
never invent a citation, and distinguish SUPPORTED CLAIM / INTERPRETATION /
LIMITATION / uncertainty explicitly. When evidence doesn't cover the
question, the model must say so instead of guessing.
"""
from __future__ import annotations

from backend.app.models.schemas import EvidenceItem

CITATION_RULES = (
    "Citation rules:\n"
    "- Cite only using the numbered evidence blocks below, with inline [n] markers "
    "(e.g. [1], or [1][3] for multiple sources).\n"
    "- Put the marker(s) inside the sentence, immediately before its final period. Every factual "
    "sentence must carry at least one marker. Never copy an example sentence from these "
    "instructions into an answer.\n"
    "- Never invent a citation number that isn't in the evidence list.\n"
    "- Never state a fact that isn't grounded in the evidence.\n"
    "- Explicitly label each claim as a SUPPORTED CLAIM (directly stated in evidence), "
    "an INTERPRETATION (your inference from evidence), or a LIMITATION (a caveat the "
    "evidence itself raises), starting each labelled sentence or bullet with its label.\n"
    "- State the finding itself, with the specific detail (a number, name, method, dataset "
    "or reason). Do not repeat the question's wording back as the answer, and do not comment "
    "on the evidence (no 'this is directly stated', 'this is supported by', 'classified as').\n"
    "- If the evidence is insufficient to answer part or all of the question, say so "
    "plainly (\"insufficient evidence to determine X\") instead of guessing.\n"
)


# Question answering (also limitations and structured-table queries) uses plainer rules. The SUPPORTED CLAIM / INTERPRETATION
# labels above are useful for synthesis tasks, but on a factual question they were pure scaffolding: in an evaluation, answers
# that carried them scored about half as faithful as the rest, because the labels invite unsupported "interpretation"
# sentences and clutter the claims. A question asks for what the evidence says, so say only that, in its own wording.
QA_RULES = (
    "Citation rules:\n"
    "- Cite only using the numbered evidence blocks below, with inline [n] markers "
    "(e.g. [1], or [1][3] for multiple sources).\n"
    # No example sentence here on purpose: a small model sometimes copies an example sentence into its answer as if it
    # were a finding ("Method A improves recall..."), which is an unfaithful statement of its own making.
    "- Put the marker(s) inside the sentence, immediately before its final period. Every "
    "factual sentence must carry at least one marker.\n"
    "- Never invent a citation number that isn't in the evidence list.\n"
    "- First find the evidence block that names the exact subject of the question (the method, model, "
    "dataset, metric or quantity it asks about) and answer from that block. Several blocks may discuss "
    "the same paper while only one holds the fact asked for; prefer the block that names every part of "
    "the question, and do not answer with a related number from a different block.\n"
    "- Answer extractively: copy the answer nearly verbatim from the evidence sentence(s) that state it. "
    "Keep every number, unit, name and term exactly as the evidence writes it, character for character. "
    "Do not paraphrase a value into different wording; changing the evidence's own words is the most "
    "common error. When in doubt, quote the evidence sentence rather than rewording it.\n"
    "- Give the value the question asks for, copied exactly as the evidence writes it, including units "
    "and any range such as 0.856±0.011. If the evidence gives several values, give the one whose row, "
    "column or sentence matches the question's wording.\n"
    "- Match the scope of the question. Work out every entity, metric, dataset or condition the question names and answer "
    "each one; leave out anything the question did not ask for, however true or interesting it is.\n"
    "- Say only what the evidence states. Stay close to its wording: keep numbers, names and "
    "terms exactly as written, and do not add background, explanation or inference of your own.\n"
    "- Write plain sentences. Do not label them (no 'SUPPORTED CLAIM', 'INTERPRETATION' or "
    "'LIMITATION'). If the evidence itself raises a caveat, state it as an ordinary sentence.\n"
    "- Always reply with at least one complete sentence that states the answer in words. A reply "
    "that is only a citation marker is not an answer. When the evidence is a table, state the value "
    "in a full sentence that names the method, the metric and the dataset it belongs to.\n"
    "- Answer the question directly with the specific detail (a number, name, method, dataset "
    "or reason), then stop. Do not repeat the question's wording back, do not open with an "
    "introduction or close with a summary, and do not comment on the evidence (no 'this is "
    "directly stated', 'this is supported by', 'classified as').\n"
    "- Say the evidence is insufficient only after checking every evidence block; if any block states the answer, give it. "
    "If it truly does not, say so plainly (\"insufficient evidence to determine X\") instead of guessing.\n"
)


def format_evidence(evidence: list[EvidenceItem]) -> str:
    blocks = []
    for item in evidence:
        header = f"[{item.citation_id}] Paper: {item.document_title} | Page: {item.page_number} | Section: {item.section}"
        blocks.append(f"{header}\n{item.text}")
    return "\n\n".join(blocks)


def build_summarization_prompt(evidence: list[EvidenceItem]) -> tuple[str, str]:
    system = (
        "You are a biomedical literature assistant summarizing evidence retrieved from "
        "scientific papers for a researcher. Be precise, concise, and strictly evidence-grounded.\n"
        + CITATION_RULES
    )
    user = (
        "Summarize the key findings in the evidence below. Organize the summary into "
        "short paragraphs or bullet points covering objective, methods, and results where "
        "available. Cite every factual statement.\n\n"
        f"EVIDENCE:\n{format_evidence(evidence)}\n\nSUMMARY:"
    )
    return system, user


def build_comparison_prompt(evidence: list[EvidenceItem], aspect: str) -> tuple[str, str]:
    system = (
        "You are a biomedical literature assistant comparing multiple papers along a "
        f"specific aspect ('{aspect}'). Present similarities and differences neutrally, "
        "without forcing a single conclusion when the evidence disagrees.\n" + CITATION_RULES
    )
    user = (
        f"Compare the papers represented in the evidence below with respect to: {aspect}.\n"
        "Structure your answer as: (1) what each relevant paper reports, cited individually, "
        "(2) a short comparative synthesis noting agreements and disagreements.\n\n"
        f"EVIDENCE:\n{format_evidence(evidence)}\n\nCOMPARISON:"
    )
    return system, user


def build_literature_review_section_prompt(evidence: list[EvidenceItem], subtopic: str) -> tuple[str, str]:
    system = (
        "You are a biomedical literature assistant writing one section of a literature "
        "review. Write only about the given subtopic/cluster, in an academic but readable "
        "tone, fully cited.\n" + CITATION_RULES
    )
    user = (
        f"Write a literature-review subsection about: {subtopic}.\n"
        "Synthesize across the evidence items below (they come from one cluster of related "
        "papers/chunks). Keep it to 1-3 short paragraphs.\n\n"
        f"EVIDENCE:\n{format_evidence(evidence)}\n\nSUBSECTION:"
    )
    return system, user


def build_research_gap_prompt(evidence: list[EvidenceItem]) -> tuple[str, str]:
    system = (
        "You are a biomedical literature assistant identifying research gaps and open "
        "questions. Only surface gaps that are actually implied by limitations, future-work "
        "statements, or absence of evidence in the provided material.\n" + CITATION_RULES
    )
    user = (
        "Based on the evidence below, identify research gaps: questions not yet answered, "
        "methodological limitations mentioned by the authors, and populations/conditions not "
        "yet studied. Cite the evidence that implies each gap. If evidence is too sparse to "
        "responsibly infer gaps, say so.\n\n"
        f"EVIDENCE:\n{format_evidence(evidence)}\n\nRESEARCH GAPS:"
    )
    return system, user


def build_conflict_detection_prompt(evidence: list[EvidenceItem]) -> tuple[str, str]:
    system = (
        "You are a biomedical literature assistant detecting conflicting findings across "
        "papers. Present conflicts even-handedly as 'Paper A reports X [n]. Paper B reports "
        "Y [m].' and suggest possible reasons for the difference (population, methodology, "
        "dataset, sample size) without declaring one paper 'correct'.\n" + CITATION_RULES
    )
    user = (
        "Identify any conflicting or divergent findings among the evidence below. For each "
        "conflict, state what each source reports (with citations) and list plausible "
        "explanations for the discrepancy. If no conflicts are apparent, say so explicitly.\n\n"
        f"EVIDENCE:\n{format_evidence(evidence)}\n\nCONFLICTS:"
    )
    return system, user


def build_qa_prompt(question: str, evidence: list[EvidenceItem]) -> tuple[str, str]:
    system = (
        "You are a biomedical question-answering assistant. Answer strictly using the "
        "provided evidence.\n" + QA_RULES
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"EVIDENCE:\n{format_evidence(evidence)}\n\nANSWER:"
        "\nAnswer with the evidence's own words, copied verbatim; do not paraphrase."
    )
    return system, user


def build_claim_verification_prompt(claim_text: str, evidence_text: str) -> tuple[str, str]:
    system = (
        "You are a fact-checking assistant for biomedical claims. Given a single claim and "
        "the evidence it cites, classify the claim strictly, then briefly explain why.\n"
        "Respond in this exact format:\n"
        "STATUS: <SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|CONTRADICTED>\n"
        "RATIONALE: <one or two sentences>\n\n"
        "Definitions:\n"
        "- SUPPORTED: the evidence directly states the claim.\n"
        "- PARTIALLY_SUPPORTED: the evidence supports part of the claim but not all of it, "
        "or supports it with weaker/different scope.\n"
        "- UNSUPPORTED: the evidence does not address the claim at all.\n"
        "- CONTRADICTED: the evidence states something that conflicts with the claim.\n"
    )
    user = f"CLAIM: {claim_text}\n\nEVIDENCE:\n{evidence_text}\n\nClassify the claim now."
    return system, user
