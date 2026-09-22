from app.rag.prompt import build_prompt


def test_prompt_requires_evidence_only_answers_and_citations():
    prompt = build_prompt(
        "What happened?",
        [
            {
                "text": "The event happened in 2024.",
                "filename": "report.pdf",
                "page": 2,
                "chunk_id": "doc-1-0",
            }
        ],
    )

    assert "Only answer using the supplied evidence" in prompt
    assert "not enough evidence" in prompt
    assert "Kanıt 1" in prompt
    assert "Do not show internal chunk IDs" in prompt
    assert "report.pdf" in prompt
