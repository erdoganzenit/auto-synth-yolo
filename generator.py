import os
import cv2
import numpy as np
import random
import argparse
from pathlib import Path

def rotate_image_and_mask(image, angle):
    """
    Nesneyi köşeleri kesilmeden (kırpılmadan) döndürür.
    Trigonometrik matris dönüşümü ile yeni tuval boyutunu hesaplar.
    """
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)

    # Döndürme matrisini al
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Yeni sınır boyutlarını hesapla
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Merkezi yeni boyuta göre kaydır
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Görüntüyü döndür (boş kalan yerleri şeffaf yap: borderValue=(0,0,0,0))
    rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def adjust_brightness(image, factor):
    """Sadece RGB kanallarının parlaklığını değiştirir, Alfa (şeffaflık) kanalına dokunmaz."""
    bgr = image[:, :, :3].astype(np.float32)
    bgr = np.clip(bgr * factor, 0, 255).astype(np.uint8)
    return np.dstack((bgr, image[:, :, 3]))

def overlay_transparent(background, overlay, x, y):
    """
    Alfa kanalını (şeffaflık) kullanarak nesneyi arka planın üzerine pürüzsüzce yapıştırır.
    Yapıştırılan nesnenin tam piksel sınırlarını (tight bounding box) döndürür.
    """
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    # Nesne arka plan sınırlarını aşıyorsa kırp
    if x + ol_w > bg_w or y + ol_h > bg_h:
        return background, None

    # Alfa maskesini [0, 1] aralığına normalize et
    alpha_mask = overlay[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha_mask

    # Hedef bölgeyi (ROI) al
    roi = background[y:y+ol_h, x:x+ol_w]

    # Alfa birleştirme (Alpha Blending) matematiği
    for c in range(3):
        roi[:, :, c] = (alpha_mask * overlay[:, :, c] + alpha_inv * roi[:, :, c]).astype(np.uint8)

    background[y:y+ol_h, x:x+ol_w] = roi

    # Şeffaf olmayan piksellerden tam sınırlayıcı kutuyu (Tight Bounding Box) bul
    non_zero_pts = cv2.findNonZero(overlay[:, :, 3])
    if non_zero_pts is None:
        return background, None

    bx, by, bw, bh = cv2.boundingRect(non_zero_pts)
    
    # Arka plan üzerindeki gerçek koordinatlar
    abs_x = x + bx
    abs_y = y + by

    # YOLO formatı için normalize edilmiş değerler (0.0 ile 1.0 arası)
    x_center = (abs_x + (bw / 2.0)) / bg_w
    y_center = (abs_y + (bh / 2.0)) / bg_h
    norm_w = bw / bg_w
    norm_h = bh / bg_h

    return background, (x_center, y_center, norm_w, norm_h)

def generate_dataset(bg_dir, obj_dir, out_dir, count, max_objects, draw_debug):
    bg_paths = list(Path(bg_dir).glob("*.*"))
    obj_paths = list(Path(obj_dir).glob("*.png"))

    if not bg_paths:
        raise FileNotFoundError(f"Arka plan klasöründe resim bulunamadı: {bg_dir}")
    if not obj_paths:
        raise FileNotFoundError(f"Nesne klasöründe PNG dosyası bulunamadı: {obj_dir}")

    # Çıktı klasörlerini oluştur
    img_out_dir = Path(out_dir) / "images"
    lbl_out_dir = Path(out_dir) / "labels"
    debug_out_dir = Path(out_dir) / "debug"
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    if draw_debug:
        debug_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Üretim başlıyor: {count} adet sentetik veri oluşturulacak...")

    for i in range(count):
        # 1. Rastgele bir arka plan seç ve oku
        bg_path = random.choice(bg_paths)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        bg_h, bg_w = bg.shape[:2]

        num_objs = random.randint(1, max_objects)
        yolo_labels = []

        for _ in range(num_objs):
            # 2. Rastgele bir şeffaf nesne seç (4 kanallı - BGRA olarak oku)
            obj_path = random.choice(obj_paths)
            obj = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
            
            if obj is None or obj.shape[2] != 4:
                print(f"[!] Uyarı: {obj_path.name} 4 kanallı (şeffaf PNG) değil, atlanıyor.")
                continue

            # Dosya adından class_id çıkar (Örn: "0_vida.png" -> class_id = 0, yoksa 0 kabul et)
            try:
                class_id = int(obj_path.stem.split("_")[0])
            except ValueError:
                class_id = 0

            # 3. Veri Çoğaltma (Augmentation) İşlemleri
            # a) Rastgele Ölçeklendirme (Arka planın %10'u ile %35'i arasında)
            scale_ratio = random.uniform(0.10, 0.35)
            target_w = int(bg_w * scale_ratio)
            aspect_ratio = obj.shape[0] / obj.shape[1]
            target_h = int(target_w * aspect_ratio)
            
            if target_w <= 10 or target_h <= 10:
                continue
            obj = cv2.resize(obj, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # b) Rastgele Döndürme (0 - 360 derece)
            angle = random.uniform(0, 360)
            obj = rotate_image_and_mask(obj, angle)

            # c) Rastgele Parlaklık Değişimi (0.6x ile 1.4x arası)
            brightness = random.uniform(0.6, 1.4)
            obj = adjust_brightness(obj, brightness)

            # d) %30 ihtimalle hafif Gaussian Blur (hareket/odak bulanıklığı simülasyonu)
            if random.random() < 0.3:
                obj[:, :, :3] = cv2.GaussianBlur(obj[:, :, :3], (3, 3), 0)

            # Boyut kontrolü
            ol_h, ol_w = obj.shape[:2]
            if ol_w >= bg_w or ol_h >= bg_h:
                continue

            # 4. Rastgele Koordinat Seçimi ve Birleştirme
            max_x = bg_w - ol_w
            max_y = bg_h - ol_h
            rand_x = random.randint(0, max_x)
            rand_y = random.randint(0, max_y)

            bg, bbox = overlay_transparent(bg, obj, rand_x, rand_y)
            if bbox:
                yolo_labels.append((class_id, *bbox))

        # 5. Görüntüyü ve YOLO Etiketini Kaydet
        file_id = f"synth_{i:05d}"
        img_save_path = img_out_dir / f"{file_id}.jpg"
        lbl_save_path = lbl_out_dir / f"{file_id}.txt"

        cv2.imwrite(str(img_save_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        with open(lbl_save_path, "w") as f:
            for lbl in yolo_labels:
                cls_id, xc, yc, w, h = lbl
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # İsteğe bağlı: Kutuların doğru çizildiğini görmek için Debug çıktısı üret
        if draw_debug and i < 20:  # Sadece ilk 20 tanesini çiz (disk dolmasın)
            debug_img = bg.copy()
            for lbl in yolo_labels:
                cls_id, xc, yc, w, h = lbl
                x1 = int((xc - w / 2) * bg_w)
                y1 = int((yc - h / 2) * bg_h)
                x2 = int((xc + w / 2) * bg_w)
                y2 = int((yc + h / 2) * bg_h)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_img, f"Class {cls_id}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(str(debug_out_dir / f"{file_id}_debug.jpg"), debug_img)

        if (i + 1) % 50 == 0 or (i + 1) == count:
            print(f"[+] İlerleme: {i + 1}/{count} tamamlandı.")

    print(f"\n[✓] İşlem tamam! Veriler '{out_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentetik Görüntü Veri Seti ve Otomatik YOLO Etiket Üreticisi")
    parser.add_argument("--bg", type=str, default="assets/backgrounds", help="Arka plan resimlerinin klasör yolu")
    parser.add_argument("--obj", type=str, default="assets/objects", help="Şeffaf (.png) nesnelerin klasör yolu")
    parser.add_argument("--out", type=str, default="dataset", help="Çıktıların kaydedileceği klasör")
    parser.add_argument("--count", type=int, default=100, help="Üretilecek toplam görüntü sayısı")
    parser.add_argument("--max-obj", type=int, default=3, help="Bir görüntüdeki maksimum nesne sayısı")
    parser.add_argument("--debug", action="store_true", help="İlk 20 fotoğraf için kutuları çizip doğruluğu gösterir")

    args = parser.parse_args()
    generate_dataset(args.bg, args.obj, args.out, args.count, args.max_obj, args.debug)
