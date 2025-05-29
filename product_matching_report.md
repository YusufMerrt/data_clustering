
# ÜRÜN EŞLEŞTİRME PROJESİ RAPORU

## 📊 VERİ SETİ BİLGİLERİ
- Toplam ürün sayısı: 1,000
- Benzersiz küme sayısı: 333
- Kategori sayısı: 10
- Ortalama başlık uzunluğu: 54.5 karakter

## 🤖 MODEL VE YÖNTEMLERİ
- Text Embedding: Sentence-BERT (all-MiniLM-L6-v2)
- Embedding boyutu: 384
- Clustering algoritmaları: dbscan, agglomerative, kmeans, spectral, hdbscan

## 📈 PERFORMANS SONUÇLARI

### DBSCAN
- ARI: 0.0121
- NMI: 0.6752
- Silhouette: 0.1103
- Accuracy: 0.8471
- Precision: 0.0086
- Recall: 0.5285
- F1-Score: 0.0170
- Sensitivity: 0.5285
- Specificity: 0.8479

### AGGLOMERATIVE
- ARI: 0.2881
- NMI: 0.8761
- Silhouette: 0.2089
- Accuracy: 0.9936
- Precision: 0.2009
- Recall: 0.5261
- F1-Score: 0.2907
- Sensitivity: 0.5261
- Specificity: 0.9948

### KMEANS
- ARI: 0.3992
- NMI: 0.8917
- Silhouette: 0.2336
- Accuracy: 0.9965
- Precision: 0.3501
- Recall: 0.4691
- F1-Score: 0.4010
- Sensitivity: 0.4691
- Specificity: 0.9978

### SPECTRAL
- ARI: 0.2359
- NMI: 0.8634
- Silhouette: 0.1234
- Accuracy: 0.9940
- Precision: 0.1748
- Recall: 0.3753
- F1-Score: 0.2385
- Sensitivity: 0.3753
- Specificity: 0.9956

### HDBSCAN
- ARI: 0.0186
- NMI: 0.4733
- Silhouette: 0.1198
- Accuracy: 0.8395
- Precision: 0.0119
- Recall: 0.7698
- F1-Score: 0.0234
- Sensitivity: 0.7698
- Specificity: 0.8397


## 🎯 SONUÇLAR VE ÖNERİLER

### En İyi Performans Gösteren Algoritma:
**KMEANS** (F1-Score: 0.4010)


### Temel Bulgular:
1. Sentence-BERT embeddings ürün başlıkları için etkili semantik temsil sağlıyor
2. Clustering performansı ground truth ile karşılaştırıldığında tatmin edici seviyede
3. Farklı clustering algoritmaları farklı güçlü yönler sergiliyor

### Geliştirme Önerileri:
1. Hiperparametre optimizasyonu ile daha iyi sonuçlar elde edilebilir
2. Farklı embedding modelleri (e.g., multilingual models) denenebilir
3. Ensemble yöntemleri ile performans artırılabilir
4. Post-processing adımları eklenebilir

---
Rapor Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
