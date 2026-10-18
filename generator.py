import os
import cv2
import numpy as np
import random
import argparse
from pathlib import Path

def rotate_image_and_mask(image, angle):
    """Nesneyi köşeleri kırpılmadan (trigonometrik tuval genişletmesi ile) döndürür."""
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def adjust_brightness(image, factor):
    """Sadece RGB kanallarının parlaklığını değiştirir, Alfa kanalını korur."""
    bgr = image[:, :, :3].astype(np.float32)
    bgr = np.clip(bgr * factor, 0, 255).astype(np.uint8)
    return np.dstack((bgr, image[:, :, 3]))

def add_camera_noise(image):
    """
    Görüntüye gerçekçi kamera sensör gürültüsü (Gaussian Noise) veya 
    ortam bozulması (Salt & Pepper) ekler.
    """
    noise_type = random.choice(["gaussian", "salt_pepper", "poisson"])
    h, w, c = image.shape

    if noise_type == "gaussian":
        # Düşük ışıkta oluşan sensör karıncalanması (ISO noise)
        sigma = random.uniform(5, 20)
        gauss = np.random.normal(0, sigma, (h, w, c)).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return noisy

    elif noise_type == "salt_pepper":
        # Anlık piksel/aktarım bozulmaları (Tuz ve Biber gürültüsü)
        noisy = image.copy()
        prob = random.uniform(0.001, 0.006)
        rnd = np.random.rand(h, w)
        noisy[rnd < (prob / 2)] = 0      # Biber (Siyah nokta)
        noisy[rnd > 1 - (prob / 2)] = 255  # Tuz (Beyaz nokta)
        return noisy

    elif noise_type == "poisson":
        # Foton gürültüsü (Shot noise) simülasyonu
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return image

def overlay_with_shadow(background, overlay, x, y, apply_shadow=True):
    """
    1. Nesnenin gerçek sınırlarını (YOLO etiketi için) gölgeden bağımsız hesaplar.
    2. Nesnenin alfa kanalından dinamik, bulanık bir gölge üretip arka plana basar.
    3. Nesneyi gölgenin üzerine yapıştırır.
    """
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    if x + ol_w > bg_w or y + ol_h > bg_h:
        return background, None

    # --- 1. ADIM: Gölgeden Etkilenmeyen Kesin YOLO Koordinatlarını Çıkar ---
    non_zero_pts = cv2.findNonZero(overlay[:, :, 3])
    if non_zero_pts is None:
        return background, None

    bx, by, bw, bh = cv2.boundingRect(non_zero_pts)
    abs_x = x + bx
    abs_y = y + by

    x_center = (abs_x + (bw / 2.0)) / bg_w
    y_center = (abs_y + (bh / 2.0)) / bg_h
    norm_w = bw / bg_w
    norm_h = bh / bg_h
    bbox = (x_center, y_center, norm_w, norm_h)

    # --- 2. ADIM: Dinamik Gölge (Drop Shadow) Oluşturma ve Basma ---
    if apply_shadow:
        # Gölge yönü (ışık açısı) ve mesafesi
        offset_x = random.randint(-12, 12)
        offset_y = random.randint(-12, 12)
        blur_ksize = random.choice([9, 13, 17, 21])  # Tek sayı olmak zorunda
        shadow_opacity = random.uniform(0.3, 0.65)   # Gölge koyuluğu

        # Gölgenin bulanıklık taşmasını önlemek için genişletilmiş tuval (padding)
        pad = blur_ksize
        sh_mask = np.zeros((ol_h + 2 * pad, ol_w + 2 * pad), dtype=np.uint8)
        sh_mask[pad:pad+ol_h, pad:pad+ol_w] = overlay[:, :, 3]

        # Alfa maskesini bulanıklaştırarak yumuşak gölgeye dönüştür
        sh_mask = cv2.GaussianBlur(sh_mask, (blur_ksize, blur_ksize), 0)

        # Gölgenin arka plandaki koordinatları
        sh_x = x - pad + offset_x
        sh_y = y - pad + offset_y
        sh_h, sh_w = sh_mask.shape

        # Arka plan sınırlarıyla kesişimi (sınır dışına taşan gölgeyi kırp)
        bg_x1, bg_y1 = max(0, sh_x), max(0, sh_y)
        bg_x2, bg_y2 = min(bg_w, sh_x + sh_w), min(bg_h, sh_y + sh_h)

        # Gölge matrisindeki karşılık gelen koordinatlar
        sm_x1, sm_y1 = bg_x1 - sh_x, bg_y1 - sh_y
        sm_x2, sm_y2 = sm_x1 + (bg_x2 - bg_x1), sm_y1 + (bg_y2 - bg_y1)

        if bg_x2 > bg_x1 and bg_y2 > bg_y1:
            roi_sh = background[bg_y1:bg_y2, bg_x1:bg_x2]
            sub_mask = (sh_mask[sm_y1:sm_y2, sm_x1:sm_x2] / 255.0) * shadow_opacity
            
            # Siyah gölge (0,0,0) olduğu için sadece arka plan piksellerini (1 - sub_mask) ile çarpmak yeterlidir
            for c in range(3):
                roi_sh[:, :, c] = (roi_sh[:, :, c] * (1.0 - sub_mask)).astype(np.uint8)
            background[bg_y1:bg_y2, bg_x1:bg_x2] = roi_sh

    # --- 3. ADIM: Asıl Nesneyi Gölgenin Üzerine Yapıştır ---
    alpha_mask = overlay[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha_mask
    roi = background[y:y+ol_h, x:x+ol_w]

    for c in range(3):
        roi[:, :, c] = (alpha_mask * overlay[:, :, c] + alpha_inv * roi[:, :, c]).astype(np.uint8)

    background[y:y+ol_h, x:x+ol_w] = roi

    return background, bbox

def generate_dataset(bg_dir, obj_dir, out_dir, count, max_objects, draw_debug, shadow_prob, noise_prob):
    bg_paths = list(Path(bg_dir).glob("*.*"))
    obj_paths = list(Path(obj_dir).glob("*.png"))

    if not bg_paths:
        raise FileNotFoundError(f"Arka plan klasöründe resim bulunamadı: {bg_dir}")
    if not obj_paths:
        raise FileNotFoundError(f"Nesne klasöründe PNG dosyası bulunamadı: {obj_dir}")

    img_out_dir = Path(out_dir) / "images"
    lbl_out_dir = Path(out_dir) / "labels"
    debug_out_dir = Path(out_dir) / "debug"
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    if draw_debug:
        debug_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Üretim başlıyor: {count} adet sentetik veri oluşturulacak...")
    print(f"[*] Parametreler -> Gölge İhtimali: %{int(shadow_prob*100)} | Gürültü İhtimali: %{int(noise_prob*100)}")

    for i in range(count):
        bg_path = random.choice(bg_paths)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        bg_h, bg_w = bg.shape[:2]

        num_objs = random.randint(1, max_objects)
        yolo_labels = []

        for _ in range(num_objs):
            obj_path = random.choice(obj_paths)
            obj = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
            
            if obj is None or obj.shape[2] != 4:
                continue

            try:
                class_id = int(obj_path.stem.split("_")[0])
            except ValueError:
                class_id = 0

            # Ölçeklendirme
            scale_ratio = random.uniform(0.10, 0.35)
            target_w = int(bg_w * scale_ratio)
            aspect_ratio = obj.shape[0] / obj.shape[1]
            target_h = int(target_w * aspect_ratio)
            
            if target_w <= 10 or target_h <= 10:
                continue
            obj = cv2.resize(obj, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Döndürme ve Parlaklık
            obj = rotate_image_and_mask(obj, random.uniform(0, 360))
            obj = adjust_brightness(obj, random.uniform(0.6, 1.4))

            # %30 ihtimalle odak bulanıklığı
            if random.random() < 0.3:
                obj[:, :, :3] = cv2.GaussianBlur(obj[:, :, :3], (3, 3), 0)

            ol_h, ol_w = obj.shape[:2]
            if ol_w >= bg_w or ol_h >= bg_h:
                continue

            rand_x = random.randint(0, bg_w - ol_w)
            rand_y = random.randint(0, bg_h - ol_h)

            # Gölge uygulanıp uygulanmayacağına karar ver
            use_shadow = random.random() < shadow_prob
            bg, bbox = overlay_with_shadow(bg, obj, rand_x, rand_y, apply_shadow=use_shadow)
            
            if bbox:
                yolo_labels.append((class_id, *bbox))

        # Tüm nesneler yapıştırıldıktan sonra tüm sahneye kamera sensör gürültüsü ekle
        if random.random() < noise_prob:
            bg = add_camera_noise(bg)

        # Kayıt İşlemleri
        file_id = f"synth_{i:05d}"
        cv2.imwrite(str(img_out_dir / f"{file_id}.jpg"), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        with open(lbl_out_dir / f"{file_id}.txt", "w") as f:
            for lbl in yolo_labels:
                cls_id, xc, yc, w, h = lbl
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        if draw_debug and i < 20:
            debug_img = bg.copy()
            for lbl in yolo_labels:
                cls_id, xc, yc, w, h = lbl
                x1, y1 = int((xc - w / 2) * bg_w), int((yc - h / 2) * bg_h)
                x2, y2 = int((xc + w / 2) * bg_w), int((yc + h / 2) * bg_h)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_img, f"Class {cls_id}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(str(debug_out_dir / f"{file_id}_debug.jpg"), debug_img)

        if (i + 1) % 50 == 0 or (i + 1) == count:
            print(f"[+] İlerleme: {i + 1}/{count} tamamlandı.")

    print(f"\n[✓] İşlem tamam! Veriler '{out_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentetik Görüntü Veri Seti ve Otomatik YOLO Etiket Üreticisi")
    parser.add_argument("--bg", type=str, default="assets/backgrounds", help="Arka plan klasörü")
    parser.add_argument("--obj", type=str, default="assets/objects", help="Şeffaf (.png) nesne klasörü")
    parser.add_argument("--out", type=str, default="dataset", help="Çıktı klasörü")
    parser.add_argument("--count", type=int, default=100, help="Üretilecek görüntü sayısı")
    parser.add_argument("--max-obj", type=int, default=3, help="Görüntü başına maksimum nesne")
    parser.add_argument("--shadow-prob", type=float, default=0.8, help="Nesne altına gölge düşürme ihtimali (0.0 - 1.0)")
    parser.add_argument("--noise-prob", type=float, default=0.5, help="Görüntüye sensör gürültüsü ekleme ihtimali (0.0 - 1.0)")
    parser.add_argument("--debug", action="store_true", help="İlk 20 fotoğraf için doğrulama kutularını çizer")

    args = parser.parse_args()
    generate_dataset(args.bg, args.obj, args.out, args.count, args.max_obj, args.debug, args.shadow_prob, args.noise_prob)
