import io
from functools import lru_cache


class OCRUnavailableError(RuntimeError):
    """Raised when OCR is needed but the easyocr package is not installed."""


@lru_cache(maxsize=2)
def get_reader(languages):
    try:
        # Imported lazily: easyocr loads torch and its models take seconds to start.
        import easyocr
        import torch
    except ImportError as exc:
        raise OCRUnavailableError("Reading scanned pages needs the easyocr package") from exc

    # The first call downloads the text detection and recognition models (about 100 MB).
    return easyocr.Reader(list(languages), gpu=torch.cuda.is_available(), verbose=False)


def read_page_images(page, languages=("tr", "en")):
    """Return the text found by OCR in the images of a pypdf page.

    A scanned PDF page is one picture of the paper, so its text can only be recovered
    by reading the picture.
    """
    import numpy as np
    from PIL import Image

    reader = get_reader(tuple(languages))
    paragraphs = []
    for image in page.images:
        try:
            picture = Image.open(io.BytesIO(image.data)).convert("RGB")
        except Exception:
            # Some embedded formats (such as JBIG2) can't be decoded by Pillow; skip them
            # rather than fail the whole document.
            continue
        paragraphs.extend(reader.readtext(np.array(picture), detail=0, paragraph=True))
    return "\n".join(paragraphs)
