from app.rag.grounding import verify_answer


EVIDENCE = [
    {"text": "Su kayıp oranı yüzde 31'den yüzde 22'ye düştü. Proje 86 milyon liraya mal oldu."},
    {"text": "Tramvay günde 21.500 yolcu taşıyor. Kişi başı yeşil alan 5,8 metrekare."},
]


def test_supported_answer_has_no_findings():
    result = verify_answer(
        "Su kaybı %31'den %22'ye düştü [Kanıt 1]; tramvay günde 21500 yolcu taşıyor [Kanıt 2].",
        EVIDENCE,
    )

    assert result == {"citations": [1, 2], "invalid_citations": [], "unsupported_numbers": []}


def test_reports_numbers_missing_from_evidence():
    result = verify_answer("Proje 90 milyon liraya mal oldu ve 86 milyonluk kısmı... [Kanıt 1]", EVIDENCE)

    assert result["unsupported_numbers"] == ["90"]


def test_reports_citations_to_missing_evidence():
    result = verify_answer("Yeşil alan 5,8 metrekare [Kanıt 2] [Kanıt 4].", EVIDENCE)

    assert result["citations"] == [2]
    assert result["invalid_citations"] == [4]


def test_turkish_number_formats_and_question_numbers_count_as_supported():
    result = verify_answer(
        "2025 yılında kişi başı yeşil alan 5.8 metrekareydi; günlük yolcu 21.500.",
        EVIDENCE,
        question="2025 yılında kişi başı yeşil alan ne kadardı?",
    )

    assert result["unsupported_numbers"] == []
