import re


_CITATION = re.compile(r"Kan[ıi]t\s*(\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")


def verify_answer(answer, evidence, question=""):
    """Cheap checks that an answer stays within its evidence.

    Reports citations that point at evidence numbers which don't exist, and numbers in
    the answer that appear neither in the evidence nor in the question. A number the
    model made up is the most common and most harmful kind of unsupported claim, and
    unlike a paraphrase it can be checked exactly.
    """
    cited = sorted({int(number) for number in _CITATION.findall(answer)})
    valid = set(range(1, len(evidence) + 1))

    known_numbers = {
        _normalize_number(number)
        for text in [question, *(item["text"] for item in evidence)]
        for number in _NUMBER.findall(text)
    }
    unsupported_numbers = []
    for number in _NUMBER.findall(_CITATION.sub(" ", answer)):
        if _normalize_number(number) not in known_numbers and number not in unsupported_numbers:
            unsupported_numbers.append(number)

    return {
        "citations": [number for number in cited if number in valid],
        "invalid_citations": [number for number in cited if number not in valid],
        "unsupported_numbers": unsupported_numbers,
    }


def _normalize_number(number):
    # Turkish texts write 18.000 for eighteen thousand and 7,4 for seven point four.
    if _THOUSANDS.fullmatch(number):
        return number.replace(".", "")
    return number.replace(",", ".").replace(":", ".")
