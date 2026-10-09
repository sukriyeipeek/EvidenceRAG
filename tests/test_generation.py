import pytest

from app.rag.generation import LLMGenerationError, generate_answer


def test_generate_answer_returns_only_new_text_deterministically():
    calls = []

    def fake_generator(messages, **kwargs):
        calls.append((messages, kwargs))
        return [{"generated_text": "  Cevap [Kanıt 1]  "}]

    messages = [{"role": "user", "content": "Soru?"}]
    answer = generate_answer(messages, max_new_tokens=32, text_generator=fake_generator)

    assert answer == "Cevap [Kanıt 1]"
    assert len(calls) == 1
    called_messages, kwargs = calls[0]
    assert called_messages == messages
    assert kwargs["generation_config"].max_new_tokens == 32
    assert kwargs["generation_config"].do_sample is False
    assert kwargs["return_full_text"] is False


def test_generate_answer_wraps_model_errors():
    def broken_generator(messages, **kwargs):
        raise RuntimeError("model unavailable")

    with pytest.raises(LLMGenerationError, match="model unavailable"):
        generate_answer([], text_generator=broken_generator)
