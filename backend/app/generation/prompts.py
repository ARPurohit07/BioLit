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
    "- Put the marker(s) inside the sentence, immediately before its final period "
    "(e.g. 'Method A improves recall over baseline B [2].'). Every factual sentence must "
    "carry at least one marker.\n"
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
    "- Put the marker(s) inside the sentence, immediately before its final period "
    "(e.g. 'Method A improves recall over baseline B [2].'). Every factual sentence must "
    "carry at least one marker.\n"
    "- Never invent a citation number that isn't in the evidence list.\n"
    "- Say only what the evidence states. Stay close to its wording: keep numbers, names and "
    "terms exactly as written, and do not add background, explanation or inference of your own.\n"
    "- Write plain sentences. Do not label them (no 'SUPPORTED CLAIM', 'INTERPRETATION' or "
    "'LIMITATION'). If the evidence itself raises a caveat, state it as an ordinary sentence.\n"
    "- Answer the question directly with the specific detail (a number, name, method, dataset "
    "or reason), then stop. Do not repeat the question's wording back, do not open with an "
    "introduction or close with a summary, and do not comment on the evidence (no 'this is "
    "directly stated', 'this is supported by', 'classified as').\n"
    "- If the evidence is insufficient to answer part or all of the question, say so "
    "plainly (\"insufficient evidence to determine X\") instead of guessing.\n"
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
