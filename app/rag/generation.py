from functools import lru_cache

class LLMGenerationError(RuntimeError):
    """Raised when the configured local language model cannot answer."""


@lru_cache(maxsize=2)
def get_text_generator(model_name):
    from transformers import pipeline

    return pipeline("text2text-generation", model=model_name)


def generate_answer(
    prompt,
    model_name="google/flan-t5-small",
    max_new_tokens=256,
    text_generator=None,
):
    try:
        generator = text_generator or get_text_generator(model_name)
        output = generator(prompt, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"].strip()
    except Exception as exc:
        raise LLMGenerationError(f"LLM generation failed: {exc}") from exc
