"""Create experiments/eval/corpus/taranmis_sayim_tutanagi.pdf, an image-only PDF.

The page is a picture of text, like a scanned paper document, so a PDF text extractor
finds nothing in it and only OCR can read it. Run from the project root:

    python experiments/eval/make_scanned_pdf.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


OUTPUT = Path(__file__).resolve().parent / "corpus" / "taranmis_sayim_tutanagi.pdf"
FONT = "C:/Windows/Fonts/arial.ttf"

LINES = [
    ("DEPO SAYIM TUTANAĞI", 44),
    ("", 30),
    ("Kuzey Ticaret A.Ş. Ankara deposunda 12 Mart 2026 tarihinde", 30),
    ("yıl sonu sayımı yapılmıştır.", 30),
    ("", 30),
    ("Sayımda 1.240 koli kâğıt havlu ve 315 koli A4 fotokopi", 30),
    ("kâğıdı bulunmuştur.", 30),
    ("", 30),
    ("Kayıtlara göre eksik çıkan 18 koli tuvalet kâğıdı için", 30),
    ("ayrıca tutanak tutulmuştur.", 30),
    ("", 30),
    ("Sayım, depo şefi ve iki depo görevlisi tarafından yapılmıştır.", 30),
    ("", 30),
    ("Bu belge kurgusaldır; değerlendirme amacıyla hazırlanmıştır.", 24),
]


def main():
    # A4 at 150 DPI, on slightly grey paper like a real scan.
    page = Image.new("L", (1240, 1754), color=243)
    draw = ImageDraw.Draw(page)
    y = 160
    for text, size in LINES:
        draw.text((120, y), text, fill=25, font=ImageFont.truetype(FONT, size))
        y += int(size * 1.6)
    # A slight blur and tilt, as a flatbed scanner would leave.
    page = page.filter(ImageFilter.GaussianBlur(0.6)).rotate(0.4, fillcolor=243)
    page.save(OUTPUT, "PDF", resolution=150)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
