from functools import lru_cache


class LLMGenerationError(RuntimeError):
    """Raised when the configured local language model cannot answer."""


@lru_cache(maxsize=2)
def get_text_generator(model_name):
    import torch
    from transformers import pipeline

    # Half-precision weights are several times slower on CPUs without native bf16 support.
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    generator = pipeline("text-generation", model=model_name, dtype=dtype)
    # Answers should be reproducible, so drop the model's default sampling settings
    # instead of overriding them per call (which makes transformers warn on every request).
    defaults = generator.model.generation_config
    defaults.do_sample = False
    defaults.temperature = None
    defaults.top_p = None
    defaults.top_k = None
    return generator


def generate_answer(
    messages,
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=256,
    text_generator=None,
):
    try:
        from transformers import GenerationConfig

        generator = text_generator or get_text_generator(model_name)
        output = generator(
            messages,
            generation_config=GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False),
            return_full_text=False,
            clean_up_tokenization_spaces=False,
        )
        return output[0]["generated_text"].strip()
    except Exception as exc:
        raise LLMGenerationError(f"LLM generation failed: {exc}") from exc
