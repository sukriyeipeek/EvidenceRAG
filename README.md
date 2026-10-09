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
Çok dilli E5 embedding (query: / passage: ön ekleri)
    ↓
FAISS cosine similarity araması
    ↓
Relevance threshold ile kanıt seçimi
    ↓
Evidence-only prompt
    ↓
Yerel instruct LLM (Qwen2.5, chat formatı)
    ↓
Yanıt ve kaynak chunk'ları
```

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
| `MAX_NEW_TOKENS` | `256` | LLM yanıt uzunluğu sınırı |
| `MAX_UPLOAD_MB` | `20` | Yüklenebilecek en büyük doküman boyutu; aşan dosyalar `413` ile reddedilir |
| `INDEX_DIR` | `data/index` | Kalıcı index'in tutulduğu klasör (göreli yollar proje köküne göre) |

`RELEVANCE_THRESHOLD`, normalize edilmiş FAISS inner product değeridir; bu nedenle cosine similarity olarak yorumlanır. Eşik embedding modeline özgüdür: E5 modelleri skorları dar bir aralıkta (yaklaşık 0.75–0.90) üretir. 0.80 değeri `data/raw/ornek_rapor.md` üzerindeki küçük bir Türkçe soru setiyle seçildi (ilgili sorular ≥ 0.80, konu dışı sorular ≤ 0.79). Embedding modelini değiştirirseniz eşiği de yeniden ayarlayın.

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
      "text": "...",
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

Kanıt threshold değerini geçmiyorsa LLM çağrılmaz ve sistem yeterli kanıt bulunamadığını açıkça bildirir. Kanıt threshold'u geçse de soruyu yanıtlamıyorsa model `NO_EVIDENCE` işareti döndürür; pipeline bu durumda da aynı mesajı boş kaynak listesiyle döner. Böylece ret mesajı küçük modelin dil becerisine bağlı kalmaz.

## Testler

```bash
pytest -q
```

Testler chunk metadata'sını, embedding/vector retrieval akışını, prompt kısıtlarını, kaynak dönüşünü, yetersiz kanıt davranışını, ayarların environment'tan okunmasını ve API endpoint'lerini (yükleme doğrulaması ve boyut sınırı dahil) kontrol eder. Gerçek embedding modeli veya LLM yüklenmez; bu fonksiyonlar testte sahte fonksiyonlarla değiştirilir, bu yüzden testler birkaç saniyede biter.

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

- Index tek proses için tasarlanmıştır; aynı `INDEX_DIR`'i kullanan birden fazla uvicorn worker'ı birbirinin değişikliklerini görmez.
- Her yükleme ve silmede index dosyanın tamamı yeniden yazılır; bu, birkaç bin dokümana kadar sorun olmaz.
- PDF metin çıkarma, taranmış görüntüler için OCR yapmaz.
- Şu anda yalnızca PDF, TXT ve Markdown desteklenir.
- LLM yanıtının verilen kanıta dayalı olması prompt ve retrieval threshold ile teşvik edilir; ayrıca claim-level doğrulama katmanı henüz yoktur.
