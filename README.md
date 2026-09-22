# EvidenceRAG

EvidenceRAG, kullanıcının yüklediği dokümanlardan kanıt parçaları getirerek soruları yanıtlayan basit bir Retrieval-Augmented Generation (RAG) uygulamasıdır. Yanıt üretirken yalnızca retrieval sonucundaki parçalar kullanılır ve kullanılan chunk kaynakları yanıtla birlikte döndürülür.

## Nasıl çalışır?

```text
PDF / TXT / MD
    ↓
Metin çıkarma ve temizleme
    ↓
Overlap'li chunk'lar
    ↓
Sentence Transformers embedding
    ↓
FAISS cosine similarity araması
    ↓
Relevance threshold ile kanıt seçimi
    ↓
Evidence-only prompt
    ↓
Yerel Transformers LLM
    ↓
Yanıt ve kaynak chunk'ları
```

PDF sayfa numaraları; dosya adı, document id, source ve chunk id ile birlikte metadata olarak korunur. Vector store şu an süreç belleğindedir; uygulama yeniden başlatıldığında dokümanlar yeniden yüklenmelidir.

## Kurulum

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

İlk embedding ve LLM kullanımı, ilgili Hugging Face modellerini indirir. Model adlarını veya RAG ayarlarını environment variable ile değiştirebilirsiniz:

| Değişken | Varsayılan | Açıklama |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence Transformers embedding modeli |
| `LLM_MODEL_NAME` | `google/flan-t5-small` | `text2text-generation` görevini destekleyen yerel model |
| `CHUNK_SIZE` | `800` | Chunk karakter uzunluğu |
| `CHUNK_OVERLAP` | `100` | Ardışık chunk'lar arasındaki karakter overlap'i |
| `TOP_K` | `5` | Retrieval sonucundaki maksimum chunk sayısı |
| `RELEVANCE_THRESHOLD` | `0.35` | Normalize embedding'lerde cosine similarity alt sınırı |
| `MAX_NEW_TOKENS` | `256` | LLM yanıt uzunluğu sınırı |

`RELEVANCE_THRESHOLD`, normalize edilmiş FAISS inner product değeridir; bu nedenle cosine similarity olarak yorumlanır. Değer, kanıtın yeterince benzer olmadığını düşündüğünüz durumda environment variable ile açıkça değiştirilebilir.

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

Desteklenen formatlar: `.pdf`, `.txt`, `.md`.

### Soru sorma

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Bu dokümanın ana sonucu nedir?"}'
```

Yanıt biçimi:

```json
{
  "answer": "... [document-id-0]",
  "sources": [
    {
      "filename": "example.pdf",
      "document_id": "document-id",
      "page": 1,
      "source": "example.pdf",
      "chunk_id": "document-id-0",
      "score": 0.72
    }
  ]
}
```

Kanıt threshold değerini geçmiyorsa LLM çağrılmaz ve sistem yeterli kanıt bulunamadığını açıkça bildirir.

## Testler

```bash
pytest -q
```

Testler chunk metadata'sını, embedding/vector retrieval akışını, prompt kısıtlarını, kaynak dönüşünü ve yetersiz kanıt davranışını kontrol eder. Gerçek LLM çağrısı yapılmaz; generation fonksiyonu testte sahte bir fonksiyonla değiştirilir.

## Proje yapısı

```text
app/
  rag/
    chunking.py
    config.py
    document_loader.py
    embedding.py
    generation.py
    pipeline.py
    prompt.py
    retrieval.py
    vector_store.py
static/
  app.js
  index.html
  styles.css
main.py
tests/
requirements.txt
```

## Bilinen sınırlamalar

- Vector store kalıcı değildir ve tek proses belleğinde tutulur.
- PDF metin çıkarma, taranmış görüntüler için OCR yapmaz.
- Şu anda yalnızca PDF, TXT ve Markdown desteklenir.
- LLM yanıtının verilen kanıta dayalı olması prompt ve retrieval threshold ile teşvik edilir; ayrıca claim-level doğrulama katmanı henüz yoktur.
