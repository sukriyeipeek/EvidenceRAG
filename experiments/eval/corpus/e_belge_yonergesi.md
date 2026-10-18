# Kuzey Ticaret A.Ş. e-Belge Uygulama Yönergesi

Bu yönerge kurgusaldır; değerlendirme amacıyla hazırlanmıştır ve gerçek mevzuatı yansıtmaz.

## 1. Amaç ve kapsam

Bu yönerge, Kuzey Ticaret A.Ş. bünyesinde düzenlenen elektronik belgelerin hazırlanması, iptali, itirazı ve saklanmasına ilişkin şirket içi kuralları belirler. Yönerge muhasebe birimi, satış birimi ve bayi destek ekibi çalışanlarını kapsar. Yönergede yer almayan durumlar için muhasebe müdürlüğünün yazılı görüşü alınır.

## 2. e-Arşiv fatura

### 2.1 Düzenleme

e-Arşiv fatura, e-Fatura kullanıcısı olmayan alıcılara düzenlenir. Fatura, satış işleminin tamamlandığı gün içinde düzenlenmeli ve alıcıya e-posta ile iletilmelidir. Toplu satışlarda faturalar en geç ertesi iş günü saat 12:00'ye kadar düzenlenir.

### 2.2 İptal

Düzenlenen bir e-Arşiv fatura, düzenleme tarihinden itibaren 7 gün içinde iptal edilebilir. Alıcı ödemeyi yapmışsa fatura iptal edilmez; bunun yerine iade faturası düzenlenir. İptal işlemi yalnızca muhasebe birimi tarafından yapılır ve her iptal için gerekçe kaydı tutulur.

### 2.3 Alıcı bilgileri

Fatura tutarına göre zorunlu alıcı bilgileri aşağıdaki tabloda gösterilmiştir.

| Fatura tutarı | Vergi kimlik numarası | Ad soyad veya unvan | Açık adres |
|---|---|---|---|
| 2.000 TL'ye kadar | İsteğe bağlı | Zorunlu | İsteğe bağlı |
| 2.000 – 15.000 TL | Zorunlu | Zorunlu | İsteğe bağlı |
| 15.000 TL üzeri | Zorunlu | Zorunlu | Zorunlu |

## 3. e-Fatura

### 3.1 Düzenleme

e-Fatura, sisteme kayıtlı alıcılara düzenlenir ve alıcının posta kutusuna otomatik olarak iletilir. Ticari fatura senaryosunda alıcı faturayı kabul veya reddedebilir; temel fatura senaryosunda bu seçenek yoktur.

### 3.2 İtiraz

Alıcı, ticari senaryoyla gönderilen bir e-Faturaya teslim tarihinden itibaren 8 gün içinde itiraz edebilir. İtiraz gelen faturalar satış biriminin onayı olmadan yeniden düzenlenmez. Süresi geçen itirazlar bayi destek ekibine yönlendirilir.

## 4. Entegrasyon hata kodları

Fatura entegrasyon servisinin döndürdüğü hata kodları ve yapılması gereken işlemler aşağıdadır.

| Kod | Anlamı | Yapılacak işlem |
|---|---|---|
| E-1001 | Alıcı vergi numarası geçersiz | Vergi numarasını cari kartta düzeltin |
| E-1002 | Alıcı sistemde kayıtlı değil | Belgeyi e-Arşiv olarak düzenleyin |
| E-1007 | Fatura tarihi gelecekte | Tarihi bugünün tarihiyle değiştirin |
| E-1012 | Para birimi tanımsız | Döviz kartını tanımlayın |
| E-1015 | Mükerrer fatura numarası | Seri numarasını kontrol edin |
| E-1019 | KDV tutarı satır toplamıyla uyuşmuyor | Satır yuvarlamalarını kontrol edin |
| E-1023 | Zorunlu alan boş | Eksik alanı tamamlayın |
| E-1028 | Belge boyutu sınırı aşıldı | Ekleri küçültün veya ayırın |
| E-1031 | İmza sertifikasının süresi dolmuş | Bilgi işlem birimine başvurun |
| E-1036 | Sunucu yanıt vermedi | 10 dakika sonra yeniden gönderin |
| E-1043 | Alıcı posta kutusu dolu | Alıcıyla iletişime geçip 24 saat sonra yeniden gönderin |
| E-1047 | İrsaliye numarası bulunamadı | İrsaliye bağlantısını kaldırın |
| E-1052 | İskonto oranı yüzde 100'ü aşıyor | İskonto satırını düzeltin |
| E-1058 | Teslim adresi eksik | Sevk adresini girin |
| E-1061 | Fatura tipi senaryoyla uyumsuz | Senaryo seçimini değiştirin |
| E-1066 | Yetkisiz kullanıcı | Kullanıcı rolünü kontrol edin |
| E-1072 | Gönderim kotası doldu | Ertesi gün yeniden deneyin |
| E-1079 | Bilinmeyen sistem hatası | Hata kaydını destek ekibine iletin |

## 5. Saklama ve arşiv

e-Arşiv ve e-Fatura belgelerinin elektronik kopyaları 10 yıl süreyle saklanır. Saklama süresi, belgenin düzenlendiği yılı izleyen takvim yılının başından itibaren hesaplanır. Arşivden belge talepleri muhasebe birimine yazılı olarak yapılır ve en geç 3 iş günü içinde karşılanır.
