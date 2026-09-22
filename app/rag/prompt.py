def build_prompt(question, evidence):
    evidence_text = "\n\n".join(
        _format_evidence(index, item) for index, item in enumerate(evidence, start=1)
    )
    return f"""You are an evidence-grounded assistant.
Only answer using the supplied evidence. Do not use outside knowledge or invent facts.
If the evidence is insufficient, say that there is not enough evidence to answer.
Cite supporting evidence in the answer using its number, for example [Kanıt 1].
Do not show internal chunk IDs in the answer.

Evidence:
{evidence_text}

Question: {question}
Answer:"""


def _format_evidence(index, item):
    page = item.get("page") if item.get("page") is not None else "n/a"
    return (
        f"Kanıt {index} | "
        f"filename={item.get('filename')} | page={page}\n{item['text']}"
    )
