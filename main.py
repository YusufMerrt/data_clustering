#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Product Matching Project - Ana Dosya
Ürün başlıklarına dayalı deep learning ile ürün eşleştirme projesi

Prd.md'deki yol haritasını takip ederek:
1. Veri analizi ve ön hazırlık
2. Text embedding (Sentence-BERT)
3. Deep learning ile benzerlik tespiti
4. Clustering algoritmaları
5. Post-processing
6. Değerlendirme metrikleri ve görselleştirme
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Deep Learning ve NLP
from sentence_transformers import SentenceTransformer
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# Clustering
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# Değerlendirme metrikleri
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
    confusion_matrix, classification_report
)

# Metin işleme
import re
import string
from collections import Counter

# Diğer
import warnings
warnings.filterwarnings('ignore')
import os # gorseller klasörü oluşturmak için

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Gerekli ek kütüphaneler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.cluster import SpectralClustering
import hdbscan
nltk.download('stopwords')
nltk.download('wordnet')

import itertools

# Add TensorFlow import and availability flags
_torch_available = True # Assuming torch was successfully imported
_tf_available = False
try:
    import tensorflow as tf
    _tf_available = True
except ImportError:
    print("TensorFlow not found. Some functionalities might be limited.")
    pass # TensorFlow is optional if not using TF models directly without sentence-transformers handling it

# Attempt to import torch and tensorflow for robust tensor handling
_torch_imported = False
try:
    import torch
    _torch_imported = True
except ImportError:
    pass

_tensorflow_imported = False
try:
    import tensorflow as tf
    _tensorflow_imported = True
except ImportError:
    pass

class ProductMatcher:
    """
    Ürün eşleştirme için ana sınıf
    """
    
    def __init__(self, data_path='sample_data.csv'):
        """
        Sınıfı başlat ve veriyi yükle
        """
        self.data_path = data_path
        self.df = None
        self.embeddings = None
        self.model = None
        self.results = {}
        self.gorseller_path = "gorseller" # Görseller klasörü yolu
        if not os.path.exists(self.gorseller_path):
            os.makedirs(self.gorseller_path)
            print(f"🖼️ '{self.gorseller_path}' klasörü oluşturuldu.")
        
    def load_and_analyze_data(self):
        """
        1. Aşama: Veri yükleme ve analiz
        """
        print("📦 Veri yükleniyor ve analiz ediliyor...")
        
        # Veriyi yükle
        self.df = pd.read_csv(self.data_path)
        
        # Sütun adlarındaki boşlukları temizle
        self.df.columns = self.df.columns.str.strip()
        
        # Temel bilgiler
        print(f"✅ Toplam kayıt sayısı: {len(self.df):,}")
        print(f"✅ Sütunlar: {list(self.df.columns)}")
        print(f"✅ Toplam benzersiz küme sayısı: {self.df['Cluster ID'].nunique():,}")
        print(f"✅ Toplam kategori sayısı: {self.df['Category ID'].nunique():,}")
        
        # Eksik veri kontrolü
        print("\n🔍 Eksik veri analizi:")
        missing_data = self.df.isnull().sum()
        print(missing_data[missing_data > 0])
        
        # Başlık uzunluklarının analizi
        self.df['title_length'] = self.df['Product Title'].str.len()
        
        print(f"\n📊 Başlık uzunluk istatistikleri:")
        print(f"Ortalama: {self.df['title_length'].mean():.1f}")
        print(f"Minimum: {self.df['title_length'].min()}")
        print(f"Maksimum: {self.df['title_length'].max()}")
        
        return self.df
    
    def preprocess_titles(self):
        """
        2. Aşama: Başlık ön işleme (İngilizce için sadeleştirilmiş)
        """
        print("\n🧽 Başlıklar temizleniyor (İngilizce)...")
        import unicodedata
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        import inflect

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        p = inflect.engine()

        # Marka/model listesi örnek (geliştirilebilir)
        known_brands = ["samsung", "apple", "lg", "bosch", "beko", "arcelik", "vestel", "siemens", "philips", "lenovo", "hp", "asus", "casper", "huawei", "xiaomi"]
        def normalize_brand(text):
            for brand in known_brands:
                if brand in text:
                    return brand + " " + text.replace(brand, "").strip()
            return text

        def number_to_words(text):
            # Sayıları yazıya çevir (ör: 2 -> two)
            def repl(m):
                try:
                    return p.number_to_words(m.group())
                except:
                    return m.group()
            return re.sub(r'\b\d+\b', repl, text)

        def clean_title(title):
            if pd.isna(title):
                return ""
            # Unicode normalizasyonu
            title = unicodedata.normalize('NFKD', str(title))
            title = title.lower()
            # Noktalama ve özel karakter temizliği
            title = re.sub(r'[^\w\s]', ' ', title)
            title = re.sub(r'\s+', ' ', title).strip()
            # Sayıları yazıya çevir
            title = number_to_words(title)
            # Marka/model normalizasyonu
            title = normalize_brand(title)
            # Stopword temizliği
            words = [w for w in title.split() if w not in stop_words and len(w) > 1]
            # İngilizce lemmatization
            words = [lemmatizer.lemmatize(w) for w in words]
            return ' '.join(words)

        self.df['clean_title'] = self.df['Product Title'].apply(clean_title)
        initial_count = len(self.df)
        self.df = self.df[self.df['clean_title'].str.len() > 0]
        final_count = len(self.df)
        print(f"✅ {initial_count - final_count} boş başlık temizlendi")
        print(f"✅ Kalan kayıt sayısı: {final_count:,}")
        
    def generate_embeddings(self, model_name='all-MiniLM-L6-v2'):
        """
        3. Aşama: Text embedding oluşturma
        Uses a Hugging Face model (all-MiniLM-L6-v2) by default.
        """
        print(f"\n🤖 Sentence Transformers ile embedding oluşturuluyor ({model_name})...")
        self.model = SentenceTransformer(model_name)
        titles = self.df['clean_title'].tolist()
        
        # SentenceTransformer's encode can return torch.Tensor, tf.Tensor, or np.ndarray
        # based on availability and the model type.
        embeddings_output = self.model.encode(
            titles, 
            convert_to_tensor=True, # Tries to return native tensor type
            show_progress_bar=True,
            batch_size=32 # Adjusted batch size, can be tuned
        )
        
        # Robustly convert to NumPy array
        if _torch_imported and isinstance(embeddings_output, torch.Tensor):
            print("🔩 Embeddings converted from PyTorch tensor to NumPy array.")
            self.embeddings = embeddings_output.cpu().numpy()
        elif _tensorflow_imported and tf.is_tensor(embeddings_output):
            print("🔩 Embeddings converted from TensorFlow tensor to NumPy array.")
            self.embeddings = embeddings_output.numpy()
        elif isinstance(embeddings_output, np.ndarray):
            print("🔩 Embeddings are already NumPy array.")
            self.embeddings = embeddings_output
        else:
            # Fallback if it's a list of lists or other format (less likely with convert_to_tensor=True)
            print("🔩 Embeddings converted from list/other to NumPy array.")
            self.embeddings = np.array(embeddings_output)
            
        print(f"✅ Embedding boyutu: {self.embeddings.shape}")
        return self.embeddings
    
    def cluster_products(self, algorithms=['dbscan', 'agglomerative', 'kmeans', 'spectral', 'hdbscan']):
        """
        4. Aşama: Clustering algoritmaları
        """
        print("\n🧩 Clustering algoritmaları uygulanıyor...")
        clustering_results = {}
        true_n_clusters = self.df['Cluster ID'].nunique()
        for algorithm in algorithms:
            print(f"\n▶️ {algorithm.upper()} uygulanıyor...")
            if algorithm == 'dbscan':
                clusterer = DBSCAN(eps=0.15, min_samples=2, metric='cosine')
                cluster_labels = clusterer.fit_predict(self.embeddings)
            elif algorithm == 'agglomerative':
                clusterer = AgglomerativeClustering(
                    n_clusters=true_n_clusters,
                    metric='cosine',
                    linkage='average'
                )
                cluster_labels = clusterer.fit_predict(self.embeddings)
            elif algorithm == 'kmeans':
                clusterer = KMeans(
                    n_clusters=true_n_clusters,
                    random_state=42,
                    n_init=10
                )
                cluster_labels = clusterer.fit_predict(self.embeddings)
            elif algorithm == 'spectral':
                clusterer = SpectralClustering(
                    n_clusters=true_n_clusters,
                    affinity='nearest_neighbors',
                    random_state=42
                )
                cluster_labels = clusterer.fit_predict(self.embeddings)
            elif algorithm == 'hdbscan':
                clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
                cluster_labels = clusterer.fit_predict(self.embeddings)
            clustering_results[algorithm] = cluster_labels
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"✅ Bulunan küme sayısı: {n_clusters}")
        self.results['clustering'] = clustering_results
        return clustering_results
    
    def evaluate_performance(self):
        """
        5. Aşama: Model performansını değerlendirme
        """
        print("\n📊 Model performansı değerlendiriliyor...")
        
        evaluation_results = {}
        true_labels = self.df['Cluster ID'].values
        
        for algorithm, predicted_labels in self.results['clustering'].items():
            print(f"\n🔍 {algorithm.upper()} Değerlendirmesi:")
            
            # Clustering metrikleri
            ari = adjusted_rand_score(true_labels, predicted_labels)
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            
            try:
                silhouette = silhouette_score(self.embeddings, predicted_labels)
            except:
                silhouette = 0.0
            
            # Binary classification olarak değerlendirme için pairwise hesaplama
            # Bu adım memory-intensive olabilir, sample alıyoruz
            sample_size = min(5000, len(self.df))
            sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
            
            y_true_pairs = []
            y_pred_pairs = []
            
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    idx_i, idx_j = sample_indices[i], sample_indices[j]
                    
                    # True labels: aynı cluster'da mı?
                    true_same = (true_labels[idx_i] == true_labels[idx_j])
                    pred_same = (predicted_labels[idx_i] == predicted_labels[idx_j])
                    
                    y_true_pairs.append(int(true_same))
                    y_pred_pairs.append(int(pred_same))
            
            # Binary classification metrikleri
            accuracy = accuracy_score(y_true_pairs, y_pred_pairs)
            precision = precision_score(y_true_pairs, y_pred_pairs, average='binary')
            recall = recall_score(y_true_pairs, y_pred_pairs, average='binary')
            f1 = f1_score(y_true_pairs, y_pred_pairs, average='binary')
            
            # Sensitivity (Recall) ve Specificity hesaplama
            tn, fp, fn, tp = confusion_matrix(y_true_pairs, y_pred_pairs).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results = {
                'ARI': ari,
                'NMI': nmi,
                'Silhouette': silhouette,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Sensitivity': sensitivity,
                'Specificity': specificity
            }
            
            evaluation_results[algorithm] = results
            
            # Sonuçları yazdır
            print(f"  📈 ARI: {ari:.4f}")
            print(f"  📈 NMI: {nmi:.4f}")
            print(f"  📈 Silhouette: {silhouette:.4f}")
            print(f"  📈 Accuracy: {accuracy:.4f}")
            print(f"  📈 Precision: {precision:.4f}")
            print(f"  📈 Recall (Sensitivity): {recall:.4f}")
            print(f"  📈 F1-Score: {f1:.4f}")
            print(f"  📈 Specificity: {specificity:.4f}")
        
        self.results['evaluation'] = evaluation_results
        return evaluation_results
    
    def visualize_results(self):
        """
        6. Aşama: Sonuçları görselleştirme
        """
        print("\n🎨 Sonuçlar görselleştiriliyor...")
        
        # 1. Embedding'leri 2D'ye indirgeme (UMAP)
        print("📊 UMAP ile embedding görselleştirmesi...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(self.embeddings)
        
        # 2. t-SNE ile de görselleştirme
        print("📊 t-SNE ile embedding görselleştirmesi...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        # Büyük veri setleri için sample alıyoruz
        sample_size = min(3000, len(self.embeddings))
        sample_indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
        embeddings_tsne = tsne.fit_transform(self.embeddings[sample_indices])
        
        # Görselleştirme fonksiyonlarını çağır
        self._plot_embedding_visualization(embeddings_2d, 'UMAP')
        self._plot_tsne_visualization(embeddings_tsne, sample_indices, 't-SNE')
        self._plot_performance_comparison()
        self._plot_cluster_distribution()
        self._plot_confusion_matrices()
        
    def _plot_embedding_visualization(self, embeddings_2d, method_name):
        """UMAP/t-SNE görselleştirmesi"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Gerçek kümeler
        scatter1 = axes[0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], 
            c=self.df['Cluster ID'], 
            cmap='tab20', 
            alpha=0.6, 
            s=50
        )
        axes[0].set_title(f'{method_name} - Gerçek Kümeler', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'{method_name} 1')
        axes[0].set_ylabel(f'{method_name} 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Tahmin edilen kümeler (DBSCAN)
        if 'dbscan' in self.results['clustering']:
            scatter2 = axes[1].scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1], 
                c=self.results['clustering']['dbscan'], 
                cmap='tab20', 
                alpha=0.6, 
                s=50
            )
            axes[1].set_title(f'{method_name} - DBSCAN Tahminleri', fontsize=14, fontweight='bold')
            axes[1].set_xlabel(f'{method_name} 1')
            axes[1].set_ylabel(f'{method_name} 2')
            plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.gorseller_path, f'embedding_visualization_{method_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_tsne_visualization(self, embeddings_tsne, sample_indices, method_name):
        """t-SNE görselleştirmesi (sample verilerle)"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        sample_true_labels = self.df['Cluster ID'].iloc[sample_indices]
        
        # Gerçek kümeler
        scatter1 = axes[0].scatter(
            embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
            c=sample_true_labels, 
            cmap='tab20', 
            alpha=0.6, 
            s=50
        )
        axes[0].set_title(f'{method_name} - Gerçek Kümeler (Sample)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'{method_name} 1')
        axes[0].set_ylabel(f'{method_name} 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Tahmin edilen kümeler
        if 'dbscan' in self.results['clustering']:
            sample_pred_labels = np.array(self.results['clustering']['dbscan'])[sample_indices]
            scatter2 = axes[1].scatter(
                embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                c=sample_pred_labels, 
                cmap='tab20', 
                alpha=0.6, 
                s=50
            )
            axes[1].set_title(f'{method_name} - DBSCAN Tahminleri (Sample)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel(f'{method_name} 1')
            axes[1].set_ylabel(f'{method_name} 2')
            plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.gorseller_path, f'embedding_visualization_{method_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_performance_comparison(self):
        """Performans karşılaştırma grafikleri"""
        evaluation_results = self.results['evaluation']
        
        # Metrikleri düzenle
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity', 'ARI', 'NMI']
        algorithms = list(evaluation_results.keys())
        
        # Performans karşılaştırma heatmap
        performance_matrix = []
        for algorithm in algorithms:
            row = [evaluation_results[algorithm][metric] for metric in metrics]
            performance_matrix.append(row)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            performance_matrix, 
            xticklabels=metrics, 
            yticklabels=[alg.upper() for alg in algorithms],
            annot=True, 
            fmt='.3f', 
            cmap='RdYlBu_r',
            center=0.5
        )
        plt.title('Algoritma Performans Karşılaştırması', fontsize=16, fontweight='bold')
        plt.xlabel('Metrikler')
        plt.ylabel('Algoritmalar')
        plt.tight_layout()
        plt.savefig(os.path.join(self.gorseller_path, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Bar plot karşılaştırma
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[alg][metric] for alg in algorithms]
            axes[i].bar(algorithms, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(algorithms)])
            axes[i].set_title(f'{metric}', fontweight='bold')
            axes[i].set_ylabel('Skor')
            axes[i].set_ylim(0, 1)
            
            # Değerleri çubukların üstüne yaz
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.gorseller_path, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_cluster_distribution(self):
        """Küme dağılım grafikleri"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gerçek küme dağılımı
        cluster_counts = self.df['Cluster ID'].value_counts().head(20)
        axes[0, 0].bar(range(len(cluster_counts)), cluster_counts.values)
        axes[0, 0].set_title('Gerçek Küme Dağılımı (Top 20)', fontweight='bold')
        axes[0, 0].set_xlabel('Küme ID')
        axes[0, 0].set_ylabel('Ürün Sayısı')
        
        # Tahmin edilen küme dağılımları
        for i, (algorithm, predicted_labels) in enumerate(self.results['clustering'].items()):
            if i >= 3:  # Maksimum 3 algoritma göster
                break
                
            unique, counts = np.unique(predicted_labels, return_counts=True)
            # Noise label'ı (-1) varsa ayır
            if -1 in unique:
                noise_idx = np.where(unique == -1)[0][0]
                noise_count = counts[noise_idx]
                unique = np.delete(unique, noise_idx)
                counts = np.delete(counts, noise_idx)
            else:
                noise_count = 0
            
            # En büyük 20 kümeyi göster
            sorted_indices = np.argsort(counts)[::-1][:20]
            top_counts = counts[sorted_indices]
            
            row, col = (0, 1) if i == 0 else ((1, 0) if i == 1 else (1, 1))
            axes[row, col].bar(range(len(top_counts)), top_counts)
            title = f'{algorithm.upper()} Küme Dağılımı (Top 20)'
            if noise_count > 0:
                title += f'\nNoise: {noise_count} ürün'
            axes[row, col].set_title(title, fontweight='bold')
            axes[row, col].set_xlabel('Küme Index')
            axes[row, col].set_ylabel('Ürün Sayısı')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.gorseller_path, 'cluster_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrices(self):
        """Confusion matrix görselleştirmeleri"""
        # Sample bir confusion matrix hesapla (pairwise comparison için)
        sample_size = min(1000, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        
        true_labels = self.df['Cluster ID'].iloc[sample_indices].values
        
        n_algorithms = len(self.results['clustering'])
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 5))
        
        if n_algorithms == 1:
            axes = [axes]
        
        for i, (algorithm, all_predicted_labels) in enumerate(self.results['clustering'].items()):
            predicted_labels = np.array(all_predicted_labels)[sample_indices]
            
            # Pairwise comparison için binary labels oluştur
            y_true_pairs = []
            y_pred_pairs = []
            
            for j in range(min(500, sample_size)):  # Daha küçük sample
                for k in range(j+1, min(500, sample_size)):
                    true_same = (true_labels[j] == true_labels[k])
                    pred_same = (predicted_labels[j] == predicted_labels[k])
                    
                    y_true_pairs.append(int(true_same))
                    y_pred_pairs.append(int(pred_same))
            
            cm = confusion_matrix(y_true_pairs, y_pred_pairs)
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[i],
                xticklabels=['Farklı Ürün', 'Aynı Ürün'],
                yticklabels=['Farklı Ürün', 'Aynı Ürün']
            )
            axes[i].set_title(f'{algorithm.upper()}\nConfusion Matrix', fontweight='bold')
            axes[i].set_xlabel('Tahmin Edilen')
            axes[i].set_ylabel('Gerçek')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.gorseller_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """
        Detaylı rapor oluşturma
        """
        print("\n📋 Detaylı rapor oluşturuluyor...")
        
        report = f"""
# ÜRÜN EŞLEŞTİRME PROJESİ RAPORU

## 📊 VERİ SETİ BİLGİLERİ
- Toplam ürün sayısı: {len(self.df):,}
- Benzersiz küme sayısı: {self.df['Cluster ID'].nunique():,}
- Kategori sayısı: {self.df['Category ID'].nunique():,}
- Ortalama başlık uzunluğu: {self.df['title_length'].mean():.1f} karakter

## 🤖 MODEL VE YÖNTEMLERİ
- Text Embedding: Sentence-BERT (all-MiniLM-L6-v2)
- Embedding boyutu: {self.embeddings.shape[1]}
- Clustering algoritmaları: {', '.join(self.results['clustering'].keys())}

## 📈 PERFORMANS SONUÇLARI

"""
        
        for algorithm, metrics in self.results['evaluation'].items():
            report += f"### {algorithm.upper()}\n"
            for metric, value in metrics.items():
                report += f"- {metric}: {value:.4f}\n"
            report += "\n"
        
        report += """
## 🎯 SONUÇLAR VE ÖNERİLER

### En İyi Performans Gösteren Algoritma:
"""
        
        # En iyi algoritmanı bul (F1-Score'a göre)
        best_algorithm = max(
            self.results['evaluation'].keys(),
            key=lambda x: self.results['evaluation'][x]['F1-Score']
        )
        
        report += f"**{best_algorithm.upper()}** (F1-Score: {self.results['evaluation'][best_algorithm]['F1-Score']:.4f})\n\n"
        
        report += """
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
"""
        
        # Raporu dosyaya kaydet
        with open('product_matching_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ Rapor 'product_matching_report.md' dosyasına kaydedildi")
        return report
    
    def tune_clustering_algorithms(self):
        """
        DBSCAN, Agglomerative ve KMeans için parametre arama ve karşılaştırmalı rapor
        """
        print("\n🔬 Kümeleme algoritmaları için parametre araması başlatılıyor...")
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
        true_labels = self.df['Cluster ID'].values
        results = []

        # 1. DBSCAN parametre grid search
        print("\n▶️ DBSCAN parametre araması:")
        for eps in np.arange(0.05, 0.25, 0.02):
            for min_samples in [2, 3, 5, 7]:
                clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                cluster_labels = clusterer.fit_predict(self.embeddings)
                n_noise = (cluster_labels == -1).sum()
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                ari = adjusted_rand_score(true_labels, cluster_labels)
                nmi = normalized_mutual_info_score(true_labels, cluster_labels)
                # F1-score (binary pairwise)
                try:
                    sample_size = min(500, len(self.df))
                    sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
                    y_true_pairs, y_pred_pairs = [], []
                    for i in range(sample_size):
                        for j in range(i+1, sample_size):
                            idx_i, idx_j = sample_indices[i], sample_indices[j]
                            y_true_pairs.append(int(true_labels[idx_i] == true_labels[idx_j]))
                            y_pred_pairs.append(int(cluster_labels[idx_i] == cluster_labels[idx_j]))
                    f1 = f1_score(y_true_pairs, y_pred_pairs, average='binary')
                except:
                    f1 = 0.0
                print(f"  eps={eps}, min_samples={min_samples} | Küme: {n_clusters}, Gürültü: {n_noise}, ARI: {ari:.3f}, NMI: {nmi:.3f}, F1: {f1:.3f}")
                results.append({'alg':'DBSCAN', 'param':f'eps={eps},min_samples={min_samples}', 'clusters':n_clusters, 'noise':n_noise, 'ARI':ari, 'NMI':nmi, 'F1':f1})

        # 2. Agglomerative Clustering için farklı linkage
        print("\n▶️ Agglomerative Clustering parametre araması:")
        for linkage in ['ward', 'complete', 'average', 'single']:
            # 'ward' sadece euclidean ile çalışır
            if linkage == 'ward':
                clusterer = AgglomerativeClustering(n_clusters=20, linkage=linkage)
            else:
                clusterer = AgglomerativeClustering(n_clusters=20, linkage=linkage, metric='cosine')
            cluster_labels = clusterer.fit_predict(self.embeddings)
            n_clusters = len(set(cluster_labels))
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            try:
                sample_size = min(500, len(self.df))
                sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
                y_true_pairs, y_pred_pairs = [], []
                for i in range(sample_size):
                    for j in range(i+1, sample_size):
                        idx_i, idx_j = sample_indices[i], sample_indices[j]
                        y_true_pairs.append(int(true_labels[idx_i] == true_labels[idx_j]))
                        y_pred_pairs.append(int(cluster_labels[idx_i] == cluster_labels[idx_j]))
                f1 = f1_score(y_true_pairs, y_pred_pairs, average='binary')
            except:
                f1 = 0.0
            print(f"  linkage={linkage} | Küme: {n_clusters}, ARI: {ari:.3f}, NMI: {nmi:.3f}, F1: {f1:.3f}")
            results.append({'alg':'Agglomerative', 'param':f'linkage={linkage}', 'clusters':n_clusters, 'noise':0, 'ARI':ari, 'NMI':nmi, 'F1':f1})

        # 3. KMeans için farklı küme sayıları
        print("\n▶️ KMeans parametre araması:")
        for n_clusters in [10, 15, 20, 30, 50, 75, 100, 150, 200]:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(self.embeddings)
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            try:
                sample_size = min(500, len(self.df))
                sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
                y_true_pairs, y_pred_pairs = [], []
                for i in range(sample_size):
                    for j in range(i+1, sample_size):
                        idx_i, idx_j = sample_indices[i], sample_indices[j]
                        y_true_pairs.append(int(true_labels[idx_i] == true_labels[idx_j]))
                        y_pred_pairs.append(int(cluster_labels[idx_i] == cluster_labels[idx_j]))
                f1 = f1_score(y_true_pairs, y_pred_pairs, average='binary')
            except:
                f1 = 0.0
            print(f"  n_clusters={n_clusters} | Küme: {n_clusters}, ARI: {ari:.3f}, NMI: {nmi:.3f}, F1: {f1:.3f}")
            results.append({'alg':'KMeans', 'param':f'n_clusters={n_clusters}', 'clusters':n_clusters, 'noise':0, 'ARI':ari, 'NMI':nmi, 'F1':f1})

        print("\n🔎 Parametre arama tamamlandı. Sonuçlar özetlendi.")
        # Sonuçları DataFrame olarak kaydet
        pd.DataFrame(results).to_csv('clustering_param_search_results.csv', index=False)
        print("💾 Sonuçlar 'clustering_param_search_results.csv' dosyasına kaydedildi.")

    def postprocess_clusters(self, min_cluster_size=5):
        print("\n🔧 Küçük kümeler birleştiriliyor...")
        for alg, labels in self.results['clustering'].items():
            unique, counts = np.unique(labels, return_counts=True)
            small_clusters = unique[counts < min_cluster_size]
            for sc in small_clusters:
                idx = np.where(labels == sc)[0]
                # En yakın büyük kümeye ata
                for i in idx:
                    # En yakın komşunun kümesini bul
                    dists = np.linalg.norm(self.embeddings[i] - self.embeddings, axis=1)
                    dists[idx] = np.inf
                    nearest = np.argmin(dists)
                    labels[i] = labels[nearest]
            self.results['clustering'][alg] = labels
        print("✅ Küçük kümeler birleştirildi.")

    def upm_style_category_threshold_analysis(self, thresholds=np.arange(0.1, 1.0, 0.1), algorithms=['kmeans', 'agglomerative', 'dbscan']):
        """
        Her kategori için ve tüm veri için farklı similarity threshold'larında clustering ve F1-score analizi yapar.
        Sonuçları UPM makalesindeki gibi çizgi grafiğiyle kaydeder.
        """
        print("\n🔬 UPM benzeri kategori & threshold analizi başlatılıyor...")
        categories = self.df['Category Label'].unique()
        results = {cat: {alg: [] for alg in algorithms} for cat in categories}
        results['ALL'] = {alg: [] for alg in algorithms}

        for cat in itertools.chain(categories, ['ALL']):
            if cat == 'ALL':
                df_cat = self.df
                emb_cat = self.embeddings
                true_labels = df_cat['Cluster ID'].values
            else:
                df_cat = self.df[self.df['Category Label'] == cat]
                if len(df_cat) < 10:
                    continue  # çok az örnekli kategori atlanır
                emb_cat = self.embeddings[df_cat.index]
                true_labels = df_cat['Cluster ID'].values
            for alg in algorithms:
                f1s = []
                for thresh in thresholds:
                    # Kümeleme algoritmasını threshold ile uygula
                    if alg == 'dbscan':
                        clusterer = DBSCAN(eps=thresh, min_samples=2, metric='cosine')
                        pred_labels = clusterer.fit_predict(emb_cat)
                    elif alg == 'agglomerative':
                        n_clusters = len(np.unique(true_labels))
                        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
                        pred_labels = clusterer.fit_predict(emb_cat)
                    elif alg == 'kmeans':
                        n_clusters = len(np.unique(true_labels))
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        pred_labels = clusterer.fit_predict(emb_cat)
                    else:
                        continue
                    # Pairwise F1-score hesapla
                    sample_size = min(500, len(df_cat))
                    sample_indices = np.random.choice(len(df_cat), sample_size, replace=False)
                    y_true_pairs, y_pred_pairs = [], []
                    for i in range(sample_size):
                        for j in range(i+1, sample_size):
                            idx_i, idx_j = sample_indices[i], sample_indices[j]
                            y_true_pairs.append(int(true_labels[idx_i] == true_labels[idx_j]))
                            y_pred_pairs.append(int(pred_labels[idx_i] == pred_labels[idx_j]))
                    if len(y_true_pairs) > 0:
                        f1 = f1_score(y_true_pairs, y_pred_pairs, average='binary')
                    else:
                        f1 = 0.0
                    f1s.append(f1)
                results[cat][alg] = f1s
        # Görselleştirme
        for cat in itertools.chain(categories, ['ALL']):
            if cat not in results or all(len(results[cat][alg]) == 0 for alg in algorithms):
                continue
            plt.figure(figsize=(10, 6))
            for alg in algorithms:
                plt.plot(thresholds, results[cat][alg], marker='o', label=alg.upper())
            plt.title(f'F1-Score vs Similarity Threshold\nKategori: {cat}')
            plt.xlabel('Similarity Threshold')
            plt.ylabel('F1-Score')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            fname = os.path.join(self.gorseller_path, f'upm_style_f1_vs_threshold_{cat.replace(" ", "_").lower()}.png')
            plt.tight_layout()
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f'✅ Grafik kaydedildi: {fname}')

def main():
    """
    Ana fonksiyon - tüm pipeline'ı çalıştır
    """
    print("🚀 ÜRÜN EŞLEŞTİRME PROJESİ BAŞLIYOR...")
    print("=" * 60)
    
    # ProductMatcher nesnesini oluştur
    matcher = ProductMatcher('sample_data.csv')
    
    try:
        # 1. Veri yükleme ve analiz
        matcher.load_and_analyze_data()
        
        # 2. Ön işleme
        matcher.preprocess_titles()
        
        # 3. Embedding oluşturma
        matcher.generate_embeddings()
        
        # --- PARAMETRE ARAMA ---
        matcher.tune_clustering_algorithms()
        
        # 4. Clustering
        matcher.cluster_products()
        
        # 5. Değerlendirme
        matcher.evaluate_performance()
        
        # 6. Görselleştirme
        matcher.visualize_results()
        
        # 7. Rapor oluşturma
        matcher.generate_report()
        
        # Post-processing
        matcher.postprocess_clusters()
        
        # UPM benzeri kategori & threshold analizi
        matcher.upm_style_category_threshold_analysis()
        
        print("\n" + "=" * 60)
        print("✅ PROJENİN TAMAMLANDI!")
        print("🎨 Görselleştirmeler ve rapor oluşturuldu")
        print("📁 Çıktı dosyaları:")
        print(f"   - {os.path.join(matcher.gorseller_path, 'embedding_visualization_umap.png')}")
        print(f"   - {os.path.join(matcher.gorseller_path, 'embedding_visualization_t-sne.png')}")
        print(f"   - {os.path.join(matcher.gorseller_path, 'performance_heatmap.png')}")
        print(f"   - {os.path.join(matcher.gorseller_path, 'performance_comparison.png')}")
        print(f"   - {os.path.join(matcher.gorseller_path, 'cluster_distributions.png')}")
        print(f"   - {os.path.join(matcher.gorseller_path, 'confusion_matrices.png')}")
        print("   - product_matching_report.md")
        print("   - clustering_param_search_results.csv")
        print(f"   - {os.path.join(matcher.gorseller_path, 'upm_style_f1_vs_threshold_*.png')}")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 