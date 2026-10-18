# auto-synth-yolo
# 🎯 Auto-Synth YOLO: Synthetic Dataset Generator & Auto-Annotator

Derin öğrenme ve nesne tespiti (Object Detection) projelerinde en büyük zaman kaybı olan **veri toplama** ve **elle etiketleme (bounding box çizme)** sürecini sıfıra indiren, tamamen yerel matematiksel matris dönüşümleriyle çalışan sentetik veri üretim aracı.

## 🚀 Çözdüğü Sorun
Binlerce fotoğraf çekip saatlerce etiketleme programlarında kutu çizmek yerine; sadece birkaç şeffaf nesne görüntüsü (`.png`) ve arka plan fotoğrafı kullanarak dakikalar içinde **YOLO formatında (%100 piksel hassasiyetinde etiketlenmiş)** binlerce eğitim verisi üretir.

## ✨ Özellikler
* **Sıfır Maliyet & Sıfır API:** Bulut servisi veya harici API kullanmaz, tamamen çevrimdışı çalışır.
* **Kayıpsız Trigonometrik Döndürme:** Nesneler 0-360 derece döndürülürken köşelerinin kırpılmaması için dinamik tuval genişletmesi (Affine Transformation) uygulanır.
* **Piksel Hassasiyetinde YOLO Etiketleme:** Döndürme sonrası oluşan boşlukları hesaba katmaz; `cv2.findNonZero` ile sadece alfa kanalı dolu olan pikselleri bularak nesneyi sıfıra sıfır saran (Tight Bounding Box) koordinatları çıkarır.
* **Otomatik Veri Çoğaltma (Augmentation):** Rastgele ölçeklendirme, parlaklık değişimi, Gaussian bulanıklaştırma ve çoklu nesne yerleştirme.

## 🛠️ Kurulum ve Kullanım

1. Depoyu klonlayın ve gereksinimleri yükleyin:
```bash
git clone [https://github.com/KULLANICI_ADIN/auto-synth-yolo.git](https://github.com/KULLANICI_ADIN/auto-synth-yolo.git)
cd auto-synth-yolo
pip install -r requirements.txt
