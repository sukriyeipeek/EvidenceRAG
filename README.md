# EvidenceRAG

EvidenceRAG, kullanıcının yüklediği dokümanlardan kanıt parçaları getirerek soruları yanıtlayan basit bir Retrieval-Augmented Generation (RAG) uygulamasıdır. Yanıt üretirken yalnızca retrieval sonucundaki parçalar kullanılır ve kullanılan chunk kaynakları yanıtla birlikte döndürülür.

## Nasıl çalışır?

```text
PDF / TXT / MD
    ↓
Metin çıkarma (metin katmanı olmayan PDF sayfalarında OCR)
    ↓
Başlık yolu eklenmiş, bölüm sınırından bölünmüş chunk'lar; tablo satırları ayrı chunk
    ↓
Çok dilli E5 embedding (query: / passage: ön ekleri)
    ↓
FAISS cosine araması + BM25 kelime araması → ağırlıklı RRF ile birleştirme
    ↓
(İsteğe bağlı) cross-encoder reranker
    ↓
Relevance threshold ile kanıt seçimi
    ↓
Evidence-only prompt
    ↓
Yerel instruct LLM (Qwen2.5, chat formatı)
    ↓
Yanıt ve kaynak chunk'ları
```

Markdown dokümanlarda her chunk, üstündeki başlıkların yoluyla başlar (`Rapor > Bütçe: ...`); böylece bir bölümün ortasından gelen chunk bile hangi doküman ve bölüme ait olduğunu söyler. Markdown tablolarının her satırı, sütun adlarıyla yazılmış ayrı bir chunk olur (`Ürün grubu: Kırtasiye; Bayi iskontosu: %14; ...`).

PDF sayfa numaraları; dosya adı, document id, source ve chunk id ile birlikte metadata olarak korunur. Index, chunk'lar ve doküman kayıtları `data/index/store.npz` dosyasına kaydedilir ve uygulama yeniden başlatıldığında otomatik yüklenir. Dosya her değişiklikte tek adımda yerine yazılır; yazma yarıda kesilirse önceki kayıt bozulmaz.

## Kurulum

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

İlk embedding ve LLM kullanımı, ilgili Hugging Face modellerini indirir (embedding ~470 MB, LLM ~3 GB). Model GPU yoksa CPU'da float32 olarak çalışır; tipik bir dizüstü işlemcide bir cevap 10–30 saniye sürer, ilk soru model yüklemesi nedeniyle daha uzun sürer. Model adlarını veya RAG ayarlarını environment variable ile ya da proje kökündeki bir `.env` dosyasıyla değiştirebilirsiniz (`.env.example` dosyasını kopyalayıp düzenleyin; ortamda zaten tanımlı değişkenler `.env`'deki değerlerden önceliklidir):

| Değişken | Varsayılan | Açıklama |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-small` | Sentence Transformers embedding modeli (Türkçe dahil çok dilli) |
| `EMBEDDING_QUERY_PREFIX` | `query: ` | Soruya eklenen ön ek; E5 dışı modeller için boş bırakın |
| `EMBEDDING_PASSAGE_PREFIX` | `passage: ` | Chunk'lara eklenen ön ek; E5 dışı modeller için boş bırakın |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Chat template'i olan, `text-generation` görevini destekleyen yerel model |
| `CHUNK_SIZE` | `800` | Chunk karakter uzunluğu |
| `CHUNK_OVERLAP` | `100` | Ardışık chunk'lar arasındaki karakter overlap'i |
| `TOP_K` | `5` | Retrieval sonucundaki maksimum chunk sayısı |
| `RELEVANCE_THRESHOLD` | `0.80` | Normalize embedding'lerde cosine similarity alt sınırı |
| `HYBRID_SEARCH` | `true` | Vektör aramasını BM25 kelime aramasıyla birleştirir |
| `RETRIEVAL_CANDIDATES` | `30` | Birleştirmeden önce her aramanın önerdiği chunk sayısı |
| `KEYWORD_WEIGHT` | `0.5` | Birleştirmede BM25 listesinin vektör listesine göre ağırlığı |
| `RERANKER_MODEL_NAME` | boş (kapalı) | Cross-encoder reranker modeli, ör. `BAAI/bge-reranker-v2-m3` (~2.3 GB, CPU'da soru başına ~5 sn) |
| `RERANK_CANDIDATES` | `20` | Reranker'ın aralarından seçtiği aday chunk sayısı |
| `OCR_LANGUAGES` | `tr,en` | Metin katmanı olmayan PDF sayfaları için EasyOCR dilleri; boş bırakılırsa OCR kapanır |
| `MAX_NEW_TOKENS` | `256` | LLM yanıt uzunluğu sınırı |
| `MAX_UPLOAD_MB` | `20` | Yüklenebilecek en büyük doküman boyutu; aşan dosyalar `413` ile reddedilir |
| `INDEX_DIR` | `data/index` | Kalıcı index'in tutulduğu klasör (göreli yollar proje köküne göre) |

`RELEVANCE_THRESHOLD`, normalize edilmiş FAISS inner product değeridir; bu nedenle cosine similarity olarak yorumlanır. Eşik embedding modeline özgüdür: E5 modelleri skorları dar bir aralıkta (yaklaşık 0.75–0.90) üretir. 0.80 değeri `data/raw/ornek_rapor.md` üzerindeki küçük bir Türkçe soru setiyle seçildi (ilgili sorular ≥ 0.80, konu dışı sorular ≤ 0.79). Embedding modelini değiştirirseniz eşiği de yeniden ayarlayın. Hibrit arama ve reranker sıralamayı değiştirir ama eşik her zaman cosine skoruna uygulanır; yalnızca kelime eşleşmesiyle alakasız bir chunk kanıt sayılmaz.

Taranmış bir PDF'in ilk yüklemesi, EasyOCR modellerini indirir (~100 MB); CPU'da bir sayfa ~8 saniyede okunur. OCR ile okunan chunk'ların metadata'sında `"ocr": true` bulunur.

## Çalıştırma

```bash
uvicorn main:app --reload
```

Sunucu başladıktan sonra web arayüzünü `http://127.0.0.1:8000/` adresinden, Swagger API ekranını ise `http://127.0.0.1:8000/docs` adresinden açabilirsiniz. Web arayüzü doküman yükleme, soru sorma, cevap gösterme ve kaynakları okunabilir kartlar halinde listeleme işlemlerini içerir.

### Sağlık kontrolü

```bash
curl http://127.0.0.1:8000/health
```

### Doküman indexleme

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@./data/example.pdf"
```

Desteklenen formatlar: `.pdf`, `.txt`, `.md`. Aynı içerikteki bir dosya (adı farklı olsa bile) tekrar yüklenirse embedding hesaplanmadan `409` döner.

### Dokümanları listeleme ve silme

```bash
curl http://127.0.0.1:8000/documents
curl -X DELETE http://127.0.0.1:8000/documents/<document_id>
```

Silinen dokümanın chunk'ları index'ten çıkarılır ve artık kanıt olarak kullanılmaz. Web arayüzündeki "Yüklü dokümanlar" listesi de aynı işlemleri yapar.

### Embedding ayarlarını değiştirmek

Kayıtlı index, hangi embedding modeli ve ön eklerle oluşturulduğunu saklar. `EMBEDDING_MODEL_NAME`, `EMBEDDING_QUERY_PREFIX` veya `EMBEDDING_PASSAGE_PREFIX` değiştirilirse eski vektörler yeni modelle karşılaştırılamaz; uygulama bu durumda sessizce yanlış sonuç döndürmek yerine açık bir hatayla başlamaz. Eski ayarlara dönün ya da `data/index/store.npz` dosyasını silip dokümanları yeniden yükleyin.

### Soru sorma

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Bu dokümanın ana sonucu nedir?"}'
```

Yanıt biçimi:

```json
{
  "answer": "... [Kanıt 1]",
  "sources": [
    {
      "evidence_number": 1,
      "cited": true,
      "text": "...",
      "filename": "example.pdf",
      "document_id": "document-id",
      "page": 1,
      "source": "example.pdf",
      "chunk_id": "document-id-0",
      "score": 0.85
    }
  ],
  "verification": {
    "citations": [1],
    "invalid_citations": [],
    "unsupported_numbers": []
  }
}
```

Kanıt threshold değerini geçmiyorsa LLM çağrılmaz ve sistem yeterli kanıt bulunamadığını açıkça bildirir. Kanıt threshold'u geçse de soruyu yanıtlamıyorsa model `NO_EVIDENCE` işareti döndürür; pipeline bu durumda da aynı mesajı boş kaynak listesiyle döner. Böylece ret mesajı küçük modelin dil becerisine bağlı kalmaz. Bu durumda `verification` alanı `null` olur.

### Cevap doğrulaması

Her cevap iki basit ama kesin kontrolden geçer (`app/rag/grounding.py`):

- **Atıflar:** Cevaptaki `[Kanıt N]` atıfları okunur. Var olan kanıtlara yapılan atıflar `citations` listesine girer ve ilgili kaynağın `cited` alanı `true` olur; olmayan bir kanıta yapılan atıflar `invalid_citations` listesine düşer.
- **Sayılar:** Cevaptaki her sayının kanıt metinlerinde veya soruda geçmesi beklenir. Geçmeyenler `unsupported_numbers` listesinde döner. Türkçe yazım farkları (`18.000` / `18000`, `7,4` / `7.4`) eşit sayılır.

Bu kontroller cevabı değiştirmez; web arayüzü sorun bulunduğunda cevabın altında uyarı gösterir. Yorumlanmış veya yeniden ifade edilmiş iddiaları yakalamazlar; uydurulmuş sayıları ve sahte atıfları yakalarlar.

## Testler

```bash
pytest -q
```

Testler chunk metadata'sını, embedding/vector retrieval akışını, prompt kısıtlarını, kaynak dönüşünü, yetersiz kanıt davranışını, ayarların environment'tan okunmasını ve API endpoint'lerini (yükleme doğrulaması ve boyut sınırı dahil) kontrol eder. Gerçek embedding modeli veya LLM yüklenmez; bu fonksiyonlar testte sahte fonksiyonlarla değiştirilir, bu yüzden testler birkaç saniyede biter.

## Değerlendirme

`experiments/eval/` altında iki Türkçe değerlendirme seti bulunur:

- `dataset.json`: birbirine benzeyen iki belediye raporu ile bir izin yönergesi ve bunlar üzerine 41 soru. Sorular dört gruptadır: cevaplanabilir (Türkçe ve İngilizce), aynı konuda olup cevabı dokümanlarda bulunmayan "tuzak" sorular ve tamamen konu dışı sorular.
- `dataset_hard.json`: ilk set hit@1 1.00'e ulaştığı için eklenen daha zor set. Kurgusal bir e-belge yönergesi (hata kodu tablosu, birbirine benzeyen iptal/itiraz bölümleri), 20 satırlık bir bayi koşulları tablosu ve yalnızca OCR ile okunabilen taranmış bir PDF (`make_scanned_pdf.py` ile üretilir) içerir; ilk setin dokümanları dikkat dağıtıcı olarak eklenir. Cevaplanabilir sorular, onları zorlaştıran şeye göre etiketlidir: `code`, `table`, `confusable`, `paraphrase`, `prose`, `scanned`.

```bash
python experiments/evaluate.py                                          # retrieval ve threshold taraması, ~1 dakika
python experiments/evaluate.py --dataset experiments/eval/dataset_hard.json
python experiments/evaluate.py --llm                                    # cevapları da üretip puanlar, CPU'da ~15-20 dakika
```

Betik uygulamayla aynı ayarları (environment ve `.env`) kullanır; `--threshold` ve `--top-k` ile geçici olarak değiştirilebilir. Ölçülenler:

- **Retrieval:** doğru chunk'ın ilk sırada (hit@1) ve ilk `TOP_K` içinde (hit@k) bulunma oranı, MRR; soru tipine göre ayrıca ve soru başına ortanca arama süresi.
- **Threshold taraması:** her eşik için korunan cevaplanabilir soru oranı ve elenen konu dışı soru oranı. Önerilen eşik, en düşük ilgili skor ile en yüksek konu dışı skorun ortasıdır.
- **Cevaplar (`--llm`):** cevaplanabilir sorularda doğru / yanlış / reddedildi; cevaplanamayan sorularda doğru ret / uydurma; doğrulama bulguları ve cevap süresi.

Her çalıştırma `experiments/results/` altına bir JSON raporu yazar (git'e eklenmez). Model, prompt veya eşik değişikliklerinden önce ve sonra çalıştırarak karşılaştırın. Yüklenemeyen dokümanlar (ör. OCR kapalıyken taranmış PDF) atlanır ve rapora yazılır; o dokümana ait sorular bulunamadı sayılır.

Zor sette her geliştirmenin retrieval etkisi (cevaplanabilir 26 soru, CPU):

| Adım | hit@1 | hit@5 | MRR | Arama süresi |
| --- | --- | --- | --- | --- |
| Başlangıç | 0.58 | 0.85 | 0.68 | 0.02 sn |
| + başlık yolu | 0.65 | 0.85 | 0.74 | 0.08 sn |
| + tablo satırları | 0.81 | 0.89 | 0.84 | 0.04 sn |
| + BM25 hibrit arama | 0.85 | 0.89 | 0.86 | 0.03 sn |
| + OCR (varsayılan ayarlar) | 0.92 | 1.00 | 0.96 | 0.02 sn |

`BAAI/bge-reranker-v2-m3` reranker'ı OCR'sız durumda hit@1'i 0.85'ten 0.89'a çıkardı (bulunabilen 23 sorunun hepsi ilk sırada) ama arama süresini soru başına ~5.5 saniyeye çıkardığı için varsayılan olarak kapalıdır.

## Proje yapısı

```text
app/
  rag/
    chunking.py         # başlık yolu, tablo satırları, chunk'lara bölme
    config.py
    document_loader.py
    embedding.py
    generation.py
    grounding.py
    keyword_index.py    # BM25 kelime araması
    ocr.py              # taranmış PDF sayfaları için EasyOCR
    pipeline.py
    prompt.py
    reranking.py        # isteğe bağlı cross-encoder reranker
    retrieval.py
    vector_store.py     # FAISS + hibrit arama
data/
  raw/ornek_rapor.md
  index/            # kalıcı index (git'e eklenmez)
experiments/
  eval/
    corpus/
    dataset.json
    dataset_hard.json
    make_scanned_pdf.py
  evaluate.py
  results/          # değerlendirme raporları (git'e eklenmez)
static/
  app.js
  index.html
  styles.css
main.py
tests/
requirements.txt
```

## Bilinen sınırlamalar

- Index tek proses için tasarlanmıştır; aynı `INDEX_DIR`'i kullanan birden fazla uvicorn worker'ı birbirinin değişikliklerini görmez.
- Her yükleme ve silmede index dosyanın tamamı yeniden yazılır; bu, birkaç bin dokümana kadar sorun olmaz.
- Başlık yolu ve tablo satırları yalnızca Markdown'da tanınır. PDF'ten çıkan metin düz olduğu için PDF başlıkları ve tabloları bu iyileştirmelerden yararlanmaz; bunun için Docling gibi yapı çıkaran bir ayrıştırıcı gerekir.
- Chunk'lama değiştiğinde (bu sürümde başlık yolu ve tablo satırları eklendi) eski kayıtlı index otomatik yenilenmez; yeni davranış için dokümanları silip yeniden yükleyin.
- OCR çıktısı hatalar içerebilir (ör. `kâğıt` yerine `k~âğıt`); sayılar genelde doğru okunur ama kontrol edilmelidir.
- Kelime araması kelimelerin ilk beş harfiyle eşleşir; çok kısa kelimeler ve beş harften sonra ayrılan kelimeler için kaba bir yaklaşımdır.
- Şu anda yalnızca PDF, TXT ve Markdown desteklenir.
- Cevap doğrulaması atıfları ve sayıları kontrol eder; sayı içermeyen iddiaların kanıta dayanıp dayanmadığını denetlemez.
- Değerlendirme setleri küçüktür (41 ve 37 soru); bir sorunun sırası değişince skor 0.04 oynar. Sonuçlar yön gösterir, kesin değildir.
