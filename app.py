import streamlit as st
import cv2
import numpy as np
import random
import shutil
import tempfile
from pathlib import Path

# generator.py içindeki çekirdek fonksiyonlarımızı içeri aktarıyoruz
from generator import (
    rotate_image_and_mask,
    adjust_brightness,
    overlay_with_shadow,
    add_camera_noise
)

st.set_page_config(
    page_title="Auto-Synth YOLO Studio",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Auto-Synth YOLO: İnteraktif Veri Üretim Stüdyosu")
st.markdown(
    "Gerçekçi gölge, kamera sensör gürültüsü ve afin matris dönüşümleriyle **%100 piksel hassasiyetinde** "
    "etiketlenmiş sentetik YOLO veri setleri üretin."
)

# --- YARDIMCI FONKSİYONLAR ---
def load_images_from_source(use_local, local_dir, uploaded_files, is_object=False):
    """Klasörden veya yüklenen dosyalardan OpenCV formatında görüntüleri okur."""
    images = []
    if use_local:
        p = Path(local_dir)
        if p.exists():
            exts = ["*.png"] if is_object else ["*.jpg", "*.jpeg", "*.png"]
            for ext in exts:
                for img_path in p.glob(ext):
                    flag = cv2.IMREAD_UNCHANGED if is_object else cv2.IMREAD_COLOR
                    img = cv2.imread(str(img_path), flag)
                    if img is not None:
                        if is_object and (len(img.shape) < 3 or img.shape[2] != 4):
                            continue
                        images.append((img_path.stem, img))
    else:
        if uploaded_files:
            for file in uploaded_files:
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                flag = cv2.IMREAD_UNCHANGED if is_object else cv2.IMREAD_COLOR
                img = cv2.imdecode(file_bytes, flag)
                if img is not None:
                    if is_object and (len(img.shape) < 3 or img.shape[2] != 4):
                        continue
                    images.append((Path(file.name).stem, img))
                file.seek(0)  # Dosya okuma imlecini başa sar
    return images

def synthesize_single_frame(bg_list, obj_list, params):
    """Belirlenen parametrelere göre tek bir sentetik kare ve YOLO etiketlerini üretir."""
    _, bg_orig = random.choice(bg_list)
    bg = bg_orig.copy()
    bg_h, bg_w = bg.shape[:2]

    num_objs = random.randint(params["min_obj"], params["max_obj"])
    yolo_labels = []

    for _ in range(num_objs):
        stem, obj_orig = random.choice(obj_list)
        obj = obj_orig.copy()

        # Dosya adından class_id çıkar (Örn: "0_vida" -> 0)
        try:
            class_id = int(stem.split("_")[0])
        except ValueError:
            class_id = 0

        # 1. Ölçeklendirme
        scale = random.uniform(params["scale_min"], params["scale_max"])
        target_w = int(bg_w * scale)
        aspect_ratio = obj.shape[0] / obj.shape[1]
        target_h = int(target_w * aspect_ratio)

        if target_w <= 10 or target_h <= 10:
            continue
        obj = cv2.resize(obj, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 2. Döndürme
        angle = random.uniform(params["rot_min"], params["rot_max"])
        obj = rotate_image_and_mask(obj, angle)

        # 3. Parlaklık
        brightness = random.uniform(params["bright_min"], params["bright_max"])
        obj = adjust_brightness(obj, brightness)

        # 4. Odak Bulanıklığı
        if random.random() < params["blur_prob"]:
            obj[:, :, :3] = cv2.GaussianBlur(obj[:, :, :3], (3, 3), 0)

        ol_h, ol_w = obj.shape[:2]
        if ol_w >= bg_w or ol_h >= bg_h:
            continue

        rand_x = random.randint(0, bg_w - ol_w)
        rand_y = random.randint(0, bg_h - ol_h)

        # 5. Gölge ve Birleştirme
        use_shadow = random.random() < params["shadow_prob"]
        bg, bbox = overlay_with_shadow(bg, obj, rand_x, rand_y, apply_shadow=use_shadow)

        if bbox:
            yolo_labels.append((class_id, *bbox))

    # 6. Kamera Sensör Gürültüsü
    if random.random() < params["noise_prob"]:
        bg = add_camera_noise(bg)

    return bg, yolo_labels

# --- SOL PANEL (SIDEBAR): PARAMETRE KONTROLLERİ ---
with st.sidebar:
    st.header("📁 1. Veri Kaynağı")
    use_local_dirs = st.checkbox("Yerel 'assets/' klasörünü kullan", value=True)
    
    uploaded_bgs, uploaded_objs = None, None
    if not use_local_dirs:
        uploaded_bgs = st.file_uploader("Arka Plan Resimleri (.jpg, .png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        uploaded_objs = st.file_uploader("Şeffaf Nesneler (4 Kanallı .png)", type=["png"], accept_multiple_files=True)

    st.divider()
    st.header("⚙️ 2. Augmentation Ayarları")
    
    col_o1, col_o2 = st.columns(2)
    min_obj = col_o1.number_input("Min Nesne", min_value=1, max_value=20, value=1)
    max_obj = col_o2.number_input("Maks Nesne", min_value=1, max_value=20, value=4)
    if min_obj > max_obj:
        min_obj = max_obj

    scale_range = st.slider("Nesne Boyut Oranı (Arka plana göre)", 0.05, 0.80, (0.12, 0.35), step=0.01)
    rot_range = st.slider("Döndürme Açısı (Derece)", 0, 360, (0, 360), step=5)
    bright_range = st.slider("Parlaklık Çarpanı", 0.3, 2.0, (0.7, 1.3), step=0.1)

    st.divider()
    st.header("🌑 3. Gerçekçilik (Gölge & Gürültü)")
    shadow_prob = st.slider("Dinamik Gölge İhtimali", 0.0, 1.0, 0.85, step=0.05)
    noise_prob = st.slider("Sensör Gürültüsü İhtimali", 0.0, 1.0, 0.50, step=0.05)
    blur_prob = st.slider("Odak Bulanıklığı (Blur) İhtimali", 0.0, 1.0, 0.25, step=0.05)

# Parametreleri tek bir sözlükte topla
params = {
    "min_obj": min_obj,
    "max_obj": max_obj,
    "scale_min": scale_range[0],
    "scale_max": scale_range[1],
    "rot_min": rot_range[0],
    "rot_max": rot_range[1],
    "bright_min": bright_range[0],
    "bright_max": bright_range[1],
    "shadow_prob": shadow_prob,
    "noise_prob": noise_prob,
    "blur_prob": blur_prob
}

# Görselleri yükle
bg_list = load_images_from_source(use_local_dirs, "assets/backgrounds", uploaded_bgs, is_object=False)
obj_list = load_images_from_source(use_local_dirs, "assets/objects", uploaded_objs, is_object=True)

# Durum bilgi çubuğu
c1, c2 = st.columns(2)
c1.info(f"🖼️ **Yüklü Arka Plan Sayısı:** {len(bg_list)}")
c2.info(f"🔩 **Yüklü Şeffaf Nesne Sayısı:** {len(obj_list)}")

if not bg_list or not obj_list:
    st.warning(
        "⚠️ İşlem yapabilmek için en az 1 arka plan ve 1 şeffaf nesne (.png) gereklidir. "
        "Lütfen `assets/backgrounds` ve `assets/objects` klasörlerine dosya ekleyin veya sol menüden dosya yükleyin."
    )
    st.stop()

# --- ANA EKRAN SEKMELERİ ---
tab_preview, tab_batch = st.tabs(["👁️ Canlı Önizleme & Kalibrasyon", "🚀 Toplu Veri Seti Üretimi (Batch)"])

with tab_preview:
    col_ctrl, col_img = st.columns([1, 2.5])
    
    with col_ctrl:
        st.subheader("Anlık Test")
        st.write("Sürgüleri değiştirdikten sonra aşağıdaki butona basarak oluşan sentetik görüntüyü ve YOLO etiketlerini anında test edin.")
        show_bbox = st.checkbox("🎯 YOLO Sınırlayıcı Kutularını Göster", value=True)
        btn_generate_sample = st.button("🎲 Rastgele Örnek Üret", type="primary", use_container_width=True)

        if "sample_data" not in st.session_state or btn_generate_sample:
            st.session_state.sample_data = synthesize_single_frame(bg_list, obj_list, params)

        synth_img, labels = st.session_state.sample_data

        st.markdown("##### 📄 Üretilen YOLO Etiketi (`.txt` içeriği)")
        if labels:
            label_str = "\n".join([f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in labels])
            st.code(label_str, language="text")
        else:
            st.write("Bu kareye nesne yerleşmedi.")

    with col_img:
        display_img = synth_img.copy()
        bg_h, bg_w = display_img.shape[:2]

        if show_bbox:
            for cls_id, xc, yc, w, h in labels:
                x1, y1 = int((xc - w / 2) * bg_w), int((yc - h / 2) * bg_h)
                x2, y2 = int((xc + w / 2) * bg_w), int((yc + h / 2) * bg_h)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_img, f"Class {cls_id}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # OpenCV BGR formatını Streamlit için RGB'ye çevir
        display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        st.image(display_rgb, caption=f"Çözünürlük: {bg_w}x{bg_h} | Tespit Edilen Nesne: {len(labels)}", use_container_width=True)

with tab_batch:
    st.subheader("📦 Toplu Veri Seti Üretimi")
    col_b1, col_b2 = st.columns(2)
    
    total_count = col_b1.number_input("Üretilecek Toplam Görüntü Sayısı", min_value=1, max_value=10000, value=100, step=50)
    out_folder_name = col_b2.text_input("Kayıt Klasörü Adı", value="dataset")

    if st.button("🚀 Üretimi Başlat", type="primary"):
        out_path = Path(out_folder_name)
        img_dir = out_path / "images"
        lbl_dir = out_path / "labels"
        
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(total_count):
            frame, frame_labels = synthesize_single_frame(bg_list, obj_list, params)
            file_id = f"synth_{i:05d}"

            cv2.imwrite(str(img_dir / f"{file_id}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(lbl_dir / f"{file_id}.txt", "w") as f:
                for cls_id, xc, yc, w, h in frame_labels:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            # İlerleme çubuğunu güncelle
            if (i + 1) % max(1, total_count // 50) == 0 or (i + 1) == total_count:
                progress_bar.progress((i + 1) / total_count)
                status_text.text(f"Üretiliyor: {i + 1} / {total_count}")

        st.success(f"✅ Harika! {total_count} adet görüntü ve YOLO etiketi `{out_path.resolve()}` dizinine kaydedildi.")

        # İndirilebilir ZIP arşivi oluştur
        with st.spinner("ZIP arşivi hazırlanıyor..."):
            tmp_zip = Path(tempfile.gettempdir()) / "yolo_synth_dataset"
            zip_file_path = shutil.make_archive(str(tmp_zip), 'zip', root_dir=str(out_path))
            
            with open(zip_file_path, "rb") as fp:
                st.download_button(
                    label="📥 Üretilen Veri Setini İndir (.zip)",
                    data=fp,
                    file_name="yolo_synthetic_dataset.zip",
                    mime="application/zip"
                )
