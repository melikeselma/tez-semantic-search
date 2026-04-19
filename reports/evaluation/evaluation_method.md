# Evaluation Method

Bu aşamada amaç arama sisteminin sonuçlarını sayısal olarak ölçmektir. Bunun için küçük bir relevance judgment dosyası hazırlanmıştır.

## 1. Gold Set

`data/evaluation/relevance_judgments.json` dosyasında her sorgu için elle belirlenmiş ilgili dataset referansları vardır.

Her kayıt şu alanlardan oluşur:

- `id`: Sorgu kimliği
- `query`: Kullanıcı sorgusu
- `category`: Sorgu kategorisi
- `source_filter`: Kaynak filtresi
- `relevant_refs`: Bu sorgu için doğru kabul edilen dataset referansları
- `notes`: Neden ilgili kabul edildiğine dair kısa açıklama

Bu dosya arama motorundan bağımsız bir beklenen sonuç listesi olarak kullanılır.

## 2. Karşılaştırılan Yöntemler

Değerlendirmede üç yöntem karşılaştırılmıştır:

- `semantic`: MiniLM embedding + FAISS index ile anlamsal arama
- `bm25`: Kelime eşleşmesine dayalı BM25 baseline
- `hybrid`: Semantic ve BM25 skorlarının normalize edilip ağırlıklı birleştirilmesi

Hybrid yöntemde varsayılan ağırlık `0.6 semantic + 0.4 BM25` olarak kullanılmıştır.

Her yöntem aynı sorgular üzerinde `Top 5` sonuç üretmiştir.

## 3. Metrikler

### Precision@5

İlk 5 sonuçtan kaç tanesi doğru?

Örnek: İlk 5 sonuçta 3 doğru dataset varsa `Precision@5 = 3 / 5 = 0.60`.

### Recall@5

Beklenen doğru datasetlerin kaç tanesi ilk 5 içinde bulundu?

Örnek: Gold set içinde 6 doğru dataset varsa ve sistem ilk 5 içinde 3 tanesini bulduysa `Recall@5 = 3 / 6 = 0.50`.

### MRR

İlk doğru sonucun kaçıncı sırada geldiğini ölçer.

İlk sonuç doğruysa `MRR = 1.0`, ikinci sonuç doğruysa `MRR = 0.5`, üçüncü sonuç doğruysa `MRR = 0.33`.

### nDCG@5

Doğru sonuçların sadece bulunup bulunmadığını değil, üst sıralarda gelip gelmediğini de ölçer.

Doğru datasetler listenin üstünde gelirse skor yükselir.

## 4. Üretilen Raporlar

Script çalışınca şu dosyalar üretilir:

- `reports/evaluation/evaluation_details.csv`
- `reports/evaluation/evaluation_summary.csv`
- `reports/evaluation/evaluation_summary.json`

`evaluation_details.csv` sorgu bazlı sonuçları gösterir.

`evaluation_summary.csv` yöntem bazlı ortalama performansı gösterir.

## 5. Mevcut Sonuç

Top 5 değerlendirmesinde mevcut sonuç:

| Method | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.500 | 0.647 | 0.861 | 0.696 |
| Hybrid | 0.533 | 0.688 | 0.958 | 0.752 |
| Semantic | 0.483 | 0.612 | 0.944 | 0.692 |

Bu sonuç Hybrid yaklaşımın mevcut küçük test setinde hem BM25'i hem de Semantic aramayı geçtiğini gösterir. Yorum olarak, BM25 kelime eşleşmesini, Semantic arama ise anlam benzerliğini yakaladığı için iki sinyalin birleşmesi daha dengeli bir sıralama üretmiştir.
