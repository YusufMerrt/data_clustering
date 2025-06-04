# 🛒 Ürün Eşleştirme Projesi (Advanced Product Matching)

Bu proje, kullanıcı tarafından girilen çeşitli ürün başlıklarını (`Product Title`), standartlaştırılmış gerçek ürün tanımlayıcıları (`Cluster Label`) ile eşleştirmek ve gruplamak için geliştirilmiştir. Proje, metin ön işleme, derin öğrenme tabanlı metin embedding (Sentence-BERT), çeşitli kümeleme algoritmaları ve kapsamlı performans değerlendirme metodolojilerini içermektedir.

## 🎯 Projenin Amacı

Temel amaç, e-ticaret platformları, fiyat karşılaştırma siteleri veya envanter sistemleri gibi kaynaklardan gelen, farklı formatlarda ve detaylarda olabilen ürün başlıklarını analiz ederek:
1.  Bu başlıkların hangi **gerçek ve standart ürüne** (veri setindeki `Cluster Label` ile temsil edilen) ait olduğunu tespit etmek.
2.  Aynı gerçek ürüne işaret eden farklı yazılımlı başlıkları **anlamsal olarak aynı kümede** toplamak.
3.  Bu eşleştirme ve kümeleme işleminin performansını, özellikle **F1 Skoru** başta olmak üzere çeşitli metriklerle `Cluster Label` referans alınarak değerlendirmek.

## ✨ Temel Özellikler

*   **Veri Yükleme ve Analiz:** `sample_data.csv` dosyasından veri yükleme, sütun temizliği, temel istatistiklerin çıkarılması.
*   **Marka ve Model Çıkarımı:** `Cluster Label`'lardan otomatik marka tespiti (`known_brands.txt`) ve `Product Title`'lardan marka/model tahmini (`extracted_brands_models.txt`).
*   **Gelişmiş Metin Ön İşleme:** Unicode normalizasyonu, küçük harfe çevirme, noktalama ve özel karakter temizliği, sayıların yazıya çevrilmesi, marka normalizasyonu, stopword temizliği ve İngilizce lemmatizasyon.
*   **Metin Embedding:** `SentenceTransformer` (varsayılan: `all-MiniLM-L6-v2`) ile ürün başlıklarından yoğun vektör temsilleri (embedding) oluşturma.
*   **Kapsamlı Kümeleme Algoritmaları:**
    *   DBSCAN
    *   Agglomerative Clustering
    *   KMeans
    *   Spectral Clustering
    *   HDBSCAN
*   **Parametre Optimizasyonu:** `tune_clustering_algorithms` ile DBSCAN, Agglomerative Clustering ve KMeans için F1 skorunu maksimize edecek parametre arama ve sonuçların `clustering_param_search_results.csv` dosyasına kaydedilmesi.
*   **Optimize Edilmiş Kümeleme:** `cluster_products` içinde, parametre arama sonuçlarına göre optimize edilmiş parametrelerle kümeleme yapılması.
*   **Performans Değerlendirme:** `Cluster Label` referans alınarak ARI, NMI, Silhouette Skoru ve ikili eşleşme bazlı (pairwise) Accuracy, Precision, Recall, F1-Skoru, Sensitivity, Specificity metriklerinin hesaplanması.
*   **Zengin Görselleştirmeler:** `gorseller` klasöründe UMAP/t-SNE ile embedding ve küme görselleştirmeleri, performans karşılaştırma heatmap ve bar grafikleri, küme dağılımları, confusion matrix'ler.
*   **Detaylı Raporlama:** `product_matching_report.md` adında, veri seti bilgileri, kullanılan yöntemler, tüm algoritmaların performans sonuçları ve en iyi F1 skorunu veren algoritmanın vurgulandığı bir rapor oluşturma.
*   **Post-processing:** Küçük kümelerin birleştirilmesi.
*   **Kategori Bazlı Threshold Analizi:** `upm_style_category_threshold_analysis` ile farklı benzerlik eşiklerinde kategori bazlı F1 skoru analizi ve grafiklerinin (`upm_style_f1_vs_threshold_*.png`) oluşturulması.

## 🛠️ Kurulum

### 1. Ön Gereksinimler
*   Python 3.8 veya üzeri.
*   Conda (önerilir) veya başka bir sanal ortam yöneticisi.

### 2. Ortam Kurulumu (Conda ile)
```bash
# Yeni bir conda ortamı oluşturun (örneğin, product_matching adında Python 3.9 ile)
conda create -n product_matching python=3.9 -y

# Ortamı aktifleştirin
conda activate product_matching
```

### 3. Bağımlılıkların Yüklenmesi
Proje ana dizinindeyken aşağıdaki komutu çalıştırın:
```bash
pip install -r requirements.txt
```
Bu komut, `requirements.txt` dosyasında listelenen tüm gerekli Python kütüphanelerini kuracaktır.

### 4. Veri Dosyası
Projenin çalışması için `sample_data.csv` (veya `ProductMatcher` sınıfını başlatırken belirttiğiniz başka bir veri dosyası) proje ana dizininde bulunmalıdır. `sample_data.csv` dosyasının en azından aşağıdaki sütunları içermesi beklenir:
*   `Product Title`: Kullanıcı tarafından girilen veya çeşitli kaynaklardan gelen ürün başlığı.
*   `Cluster Label`: Her bir ürün için standartlaştırılmış, gerçek ve "temiz" ürün adı/tanımlayıcısı. Bu sütun, performans değerlendirmesi için referans (ground truth) olarak kullanılır.
*   `Cluster ID`: (Opsiyonel, `Cluster Label` daha öncelikli) Gerçek ürün gruarını belirten bir kimlik.
*   `Category ID`: (Opsiyonel) Ürün kategorisi kimliği.
*   `Category Label`: (Opsiyonel, `upm_style_category_threshold_analysis` için kullanılır) Ürün kategorisi etiketi.

## 🚀 Projeyi Çalıştırma

Proje ana dizinindeyken (ve sanal ortamınız aktifken) aşağıdaki komutu çalıştırın:
```bash
python main.py
```
Bu komut, `main.py` içerisindeki `main()` fonksiyonunu tetikleyecek ve tüm ürün eşleştirme pipeline'ını adım adım yürütecektir. Konsolda her adım hakkında bilgi mesajları ve ilerleme çubukları göreceksiniz.

## 📁 Proje Yapısı

```
veri_odev_yusufi/
│
├── main.py                      # Ana Python betiği, ProductMatcher sınıfını ve ana iş akışını içerir
├── requirements.txt             # Proje için gerekli Python kütüphaneleri
├── README.md                    # Bu bilgilendirme dosyası
├── sample_data.csv              # Örnek veri seti (Product Title, Cluster Label vb. sütunları içermeli)
│
├── gorseller/                   # Çalıştırma sonrası oluşturulan tüm görsel dosyalar bu klasöre kaydedilir
│   ├── embedding_visualization_umap.png
│   ├── embedding_visualization_t-sne.png
│   ├── performance_heatmap.png
│   ├── performance_comparison.png
│   ├── cluster_distributions.png
│   ├── confusion_matrices.png
│   └── upm_style_f1_vs_threshold_*.png # Kategori bazlı F1 analiz grafikleri
│
├── known_brands.txt             # Cluster Label'lardan tespit edilen markaların listesi
├── extracted_brands_models.txt  # Product Title'lardan çıkarılan marka/model tahminleri ve örnekleri
├── product_matching_report.md   # Çalıştırma sonrası oluşturulan detaylı performans raporu
└── clustering_param_search_results.csv # Kümeleme algoritmaları için yapılan parametre arama sonuçları
```

## ⚙️ Metodoloji ve Fonksiyonlar (`ProductMatcher` Sınıfı)

Projenin kalbi `main.py` dosyasındaki `ProductMatcher` sınıfıdır. İşte bu sınıfın temel fonksiyonları ve işleyişleri:

1.  **`__init__(self, data_path='sample_data.csv')`**
    *   Sınıfı başlatır, veri dosyasının yolunu (`data_path`) alır.
    *   Sonuçları, embedding'leri ve modeli saklamak için dahili değişkenleri tanımlar.
    *   `gorseller` adında bir klasör yoksa oluşturur.

2.  **`extract_brand_model_from_title(self, title)`**
    *   Verilen bir `Product Title` (ürün başlığı) içinden, `self.known_brands` listesini kullanarak marka ve ardından gelen ilk birkaç kelimeyi model olarak çıkarmaya çalışır.

3.  **`load_and_analyze_data(self)`**
    *   Belirtilen `data_path`'ten CSV dosyasını `pandas` DataFrame olarak yükler.
    *   Sütun adlarındaki olası baş/son boşluklarını temizler.
    *   `Cluster Label` sütunundaki ilk kelimeleri alarak `self.known_brands` listesini oluşturur ve bu listeyi `known_brands.txt` dosyasına yazar.
    *   `Product Title` sütunundaki her başlık için `extract_brand_model_from_title` fonksiyonunu kullanarak marka/model çıkarımı yapar ve sonuçları `extracted_brands_models.txt` dosyasına örneklerle birlikte yazar.
    *   Veri seti hakkında temel istatistikler (toplam kayıt, sütunlar, benzersiz `Cluster Label` ve kategori sayısı) basar.
    *   Eksik veri analizi yapar ve başlık uzunluk istatistiklerini gösterir.

4.  **`preprocess_titles(self)`**
    *   `Product Title` sütunundaki başlıkları temizler ve standartlaştırır:
        *   Unicode normalizasyonu (NFKD).
        *   Küçük harfe çevirme.
        *   Noktalama işaretleri ve özel karakterlerin boşlukla değiştirilmesi (`re.sub(r'[^\w\s]', ' ', title)`).
        *   Fazla boşlukların tek boşluğa indirilmesi ve baş/son boşlukların temizlenmesi.
        *   Sayıların yazıya çevrilmesi (örn: "2" -> "two") (`inflect` kütüphanesi).
        *   Basit bir marka normalizasyonu (statik `known_brands` listesi ve `self.known_brands` ile geliştirilebilir).
        *   İngilizce `stopwords` (nltk) ve birden kısa kelimelerin kaldırılması.
        *   İngilizce lemmatizasyon (`WordNetLemmatizer` nltk).
    *   Temizlenmiş başlıkları `clean_title` adında yeni bir sütuna kaydeder.
    *   Temizlik sonrası boş kalan başlıkları veri setinden çıkarır.

5.  **`generate_embeddings(self, model_name='all-MiniLM-L6-v2')`**
    *   Belirtilen `model_name` (varsayılan: `all-MiniLM-L6-v2`) ile bir `SentenceTransformer` modeli yükler.
    *   `clean_title` sütunundaki temizlenmiş başlıkları kullanarak her başlık için bir embedding (yoğun vektör temsili) oluşturur.
    *   Oluşturulan embedding'leri NumPy array formatında `self.embeddings` değişkenine kaydeder.

6.  **`extract_word_combinations(self, titles)`, `evaluate_combinations(self, combinations, titles)`, `initial_clustering(self, titles, top_combinations)`, `validate_and_recluster(self, clusters, titles, embeddings)`**
    *   Bu fonksiyonlar, `cluster_products` içinde kullanılan, kelime kombinasyonlarına dayalı deneysel bir kümeleme yaklaşımının parçalarıdır. Başlıklardan 2'li ve 3'lü kelime kombinasyonları çıkarır, bunları frekans, marka içerme ve uzunluk gibi kriterlere göre skorlar, en iyi kombinasyonlara göre ilk kümeleri oluşturur ve ardından bu kümeleri embedding benzerliklerine göre doğrular ve yeniden düzenler. *(Not: `cluster_products` metodunun güncel halinde bu özel yaklaşımın yanı sıra standart kümeleme algoritmaları da optimize edilmiş parametrelerle kullanılmaktadır.)*

7.  **`cluster_products(self, algorithms=['dbscan', 'agglomerative', 'kmeans', 'spectral', 'hdbscan'])`**
    *   Belirtilen `algorithms` listesindeki her bir kümeleme algoritmasını `self.embeddings` üzerine uygular.
    *   **Optimize Edilmiş Parametreler:**
        *   **DBSCAN:** `eps=0.23`, `min_samples=2`, `metric='cosine'` (F1 optimizasyonundan).
        *   **Agglomerative Clustering:** `linkage='ward'`, `metric='euclidean'`, `n_clusters` (varsayılan olarak `Cluster Label` sayısına eşit, F1 optimizasyonundan).
        *   **KMeans:** `n_clusters=200` (F1 optimizasyonundan).
        *   **Spectral Clustering:** `n_clusters` (`Cluster Label` sayısına eşit), `affinity='nearest_neighbors'`.
        *   **HDBSCAN:** `min_cluster_size=5`, `metric='euclidean'`.
    *   Her algoritmanın ürettiği küme etiketlerini `self.results['clustering']` altında saklar.

8.  **`evaluate_performance(self)`**
    *   `self.results['clustering']` altında saklanan her kümeleme algoritmasının sonuçlarını, `self.df['Cluster Label'].values` (gerçek ürün etiketleri) ile karşılaştırarak performans metrikleri hesaplar:
        *   **Kümeleme Metrikleri:** Adjusted Rand Score (ARI), Normalized Mutual Information (NMI), Silhouette Score.
        *   **İkili Eşleşme Bazlı Metrikler (Pairwise):** Rastgele seçilen ürün çiftlerinin aynı `Cluster Label`'a sahip olup olmadığı (gerçek durum) ile aynı kümeye atanıp atanmadığı (tahmini durum) karşılaştırılarak Accuracy, Precision, Recall, F1-Score, Sensitivity (Recall ile aynı) ve Specificity hesaplanır.
    *   Hesaplanan tüm metrikleri `self.results['evaluation']` altında saklar.

9.  **`visualize_results(self)`**
    *   `_plot_embedding_visualization`: UMAP ve t-SNE kullanarak embedding'leri 2D'ye indirger ve hem gerçek `Cluster ID`'lere (görsel referans için) hem de her bir algoritmanın tahmin ettiği kümelere göre renklendirilmiş saçılım grafikleri oluşturur (`embedding_visualization_umap.png`, `embedding_visualization_t-sne.png`).
    *   `_plot_performance_comparison`: `self.results['evaluation']`'daki metrikleri kullanarak algoritmaların performansını karşılaştıran bir heatmap (`performance_heatmap.png`) ve her metrik için ayrı bar grafikler (`performance_comparison.png`) oluşturur.
    *   `_plot_cluster_distribution`: Gerçek `Cluster ID` dağılımını ve seçilen birkaç algoritmanın ürettiği küme boyutlarının dağılımını gösteren bar grafikler oluşturur (`cluster_distributions.png`).
    *   `_plot_confusion_matrices`: İkili eşleşme bazlı değerlendirme için her algoritmanın bir confusion matrix'ini oluşturur (`confusion_matrices.png`).
    *   Tüm görseller `gorseller/` klasörüne kaydedilir.

10. **`generate_report(self)`**
    *   Projenin çalışması hakkında detaylı bir Markdown raporu (`product_matching_report.md`) oluşturur. Rapor şunları içerir:
        *   Veri seti bilgileri.
        *   Kullanılan model ve yöntemler.
        *   Her kümeleme algoritması için tüm performans metrikleri.
        *   En iyi F1 skorunu elde eden algoritmanın vurgulanması.
        *   Temel bulgular ve geliştirme önerileri.

11. **`tune_clustering_algorithms(self)`**
    *   DBSCAN, Agglomerative Clustering ve KMeans için farklı parametre kombinasyonlarını deneyerek bir grid search/parametre arama işlemi yapar.
    *   Her kombinasyon için ARI, NMI ve ikili eşleşme bazlı F1 skorunu (referans olarak `Cluster ID` kullanarak, bu kısım `Cluster Label`'a güncellenebilir/paralel çalıştırılabilir) hesaplar.
    *   Sonuçları konsola basar ve `clustering_param_search_results.csv` dosyasına kaydeder. Bu dosya, `cluster_products` fonksiyonundaki algoritmaların parametrelerini optimize etmek için kullanılır.

12. **`postprocess_clusters(self, min_cluster_size=5)`**
    *   Her kümeleme algoritmasının ürettiği kümelerden, belirtilen `min_cluster_size`'dan daha az elemana sahip olan küçük kümeleri bulur.
    *   Bu küçük kümelerdeki elemanları, embedding uzayında kendilerine en yakın (gürültü olmayan) komşularının bulunduğu kümeye atayarak birleştirir.

13. **`upm_style_category_threshold_analysis(self, thresholds=np.arange(0.1, 1.0, 0.1), algorithms=['kmeans', 'agglomerative', 'dbscan'])`**
    *   Her bir ürün kategorisi (`Category Label`) ve tüm veri seti için, belirtilen kümeleme algoritmalarını farklı benzerlik eşik değerleri (DBSCAN için `eps` olarak kullanılır, diğerleri için dolaylı etki) ile çalıştırır.
    *   Her kategori, algoritma ve eşik değeri için ikili eşleşme bazlı F1 skorunu (referans olarak `Cluster ID` kullanarak, bu kısım da `Cluster Label`'a güncellenebilir/paralel çalıştırılabilir) hesaplar.
    *   Sonuçları, her kategori için F1 skorunun eşik değerine karşı değişimini gösteren çizgi grafikleri olarak (`gorseller/upm_style_f1_vs_threshold_*.png`) kaydeder.

## 💡 F1 Skoru ve Optimizasyon

Bu projede **F1 Skoru**, kullanıcı tarafından girilen `Product Title`'ların, gerçek ürün kimlikleri olan `Cluster Label`'lar ile ne kadar başarılı bir şekilde eşleştirildiğinin ana göstergelerinden biridir. Yüksek F1 skoru, modelin hem doğru ürünleri bir araya getirmede (Recall) hem de bunu yaparken farklı ürünleri karıştırmamada (Precision) başarılı olduğu anlamına gelir. `evaluate_performance` fonksiyonu, bu F1 skorunu `Cluster Label`'ları referans alarak hesaplar. `cluster_products` fonksiyonundaki algoritmaların (özellikle DBSCAN, Agglomerative, KMeans) parametreleri, `tune_clustering_algorithms` adımından elde edilen ve F1 skorunu maksimize etmeyi hedefleyen değerlerle güncellenmiştir.

## 🔧 Olası Geliştirmeler ve İpuçları
*   `preprocess_titles` içindeki `normalize_brand` fonksiyonunu, `load_and_analyze_data` içinde `Cluster Label`'lardan dinamik olarak oluşturulan `self.known_brands` listesini kullanacak şekilde güncelleyebilirsiniz.
*   `tune_clustering_algorithms` ve `upm_style_category_threshold_analysis` fonksiyonlarındaki referans etiketlerin (şu anda bazı yerlerde `Cluster ID` olabilir) tutarlı bir şekilde `Cluster Label` olmasını sağlayabilirsiniz.
*   Farklı SentenceTransformer modelleri (`all-mpnet-base-v2` gibi daha büyük modeller veya e-ticarete özel modeller) deneyerek embedding kalitesini artırabilirsiniz.
*   Metin ön işleme adımlarını (`stopwords` listesi, normalizasyon kuralları) veri setinize ve `Cluster Label`'larınızın yapısına göre daha da özelleştirebilirsiniz.

## 🐛 Sorun Giderme
*   **Python Sürümü ve Ortam Sorunları:** Projenin Python 3.8+ ve `requirements.txt` ile kurulan bir sanal ortamda çalıştırıldığından emin olun.
*   **Kütüphane İndirme Hataları (nltk):** `nltk.download('stopwords')` ve `nltk.download('wordnet')` komutları ilk çalıştırmada internet bağlantısı gerektirir. Başarısız olursa manuel olarak Python interpretöründe çalıştırılabilir.
*   **Bellek Hataları (Memory Error):** Çok büyük veri setleriyle çalışırken, `evaluate_performance` ve `visualize_results` gibi fonksiyonlardaki `sample_size` değerlerini küçültmeyi veya daha fazla RAM'e sahip bir ortamda çalışmayı düşünebilirsiniz.

---
Bu `README.md` dosyası, projenin mevcut durumu ve yetenekleri hakkında kapsamlı bir genel bakış sunar.
