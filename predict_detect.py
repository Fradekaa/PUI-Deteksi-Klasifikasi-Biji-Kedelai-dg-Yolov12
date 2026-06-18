# ==========================================
# INFERENCE YOLOv12 DETECTION
# SoyGuard AI
# ==========================================

from ultralytics import YOLO
import cv2
import os


def main():

    # ======================================
    # LOAD TRAINED MODEL
    # ======================================

    model = YOLO(
        r"D:\Kuliah\Semester 6\PUI\runs\detect\runs\detect\soybean_detect_v12\weights\best.pt"
    )

    # ======================================
    # INPUT IMAGE
    # ======================================

    image_path = input(
        "\nMasukkan path gambar : "
    )

    # Cek apakah file ada
    if not os.path.exists(image_path):

        print("\n❌ Gambar tidak ditemukan!")
        return

    # ======================================
    # PREDICT
    # ======================================

    results = model.predict(

        source=image_path,

        conf=0.25,

        imgsz=960,

        save=True,

        show=True,

        line_width=2,

        save_txt=False,

        save_conf=False

    )

    # ======================================
    # PRINT RESULT
    # ======================================

    total_soybean = 0
    total_intact = 0
    total_defect = 0

    print("\n====================================")
    print("HASIL DETEKSI")
    print("====================================\n")

    for r in results:

        boxes = r.boxes

        total_soybean = len(boxes)

        for i, box in enumerate(boxes):

            cls_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = r.names[cls_id]

            print(
                f"{i+1}. "
                f"{class_name} "
                f"({confidence*100:.2f}%)"
            )

            # ==================================
            # COUNT CLASS
            # ==================================

            if class_name == "intact_soybeans":

                total_intact += 1

            else:

                total_defect += 1

    # ======================================
    # QUALITY SCORE
    # ======================================

    if total_soybean > 0:

        quality_score = (
            total_intact /
            total_soybean
        ) * 100

    else:

        quality_score = 0

    # ======================================
    # GRADE
    # ======================================

    if quality_score >= 90:

        grade = "A"
        quality = "Sangat Layak"

    elif quality_score >= 75:

        grade = "B"
        quality = "Layak"

    elif quality_score >= 60:

        grade = "C"
        quality = "Cukup Layak"

    else:

        grade = "D"
        quality = "Tidak Layak"

    # ======================================
    # SUMMARY
    # ======================================

    print("\n====================================")
    print("RINGKASAN")
    print("====================================")

    print(f"Total Soybean : {total_soybean}")
    print(f"Total Intact  : {total_intact}")
    print(f"Total Defect  : {total_defect}")
    print(f"Quality Score : {quality_score:.2f}%")
    print(f"Grade         : {grade}")
    print(f"Quality       : {quality}")

    print("\n====================================")
    print("Hasil gambar tersimpan di:")
    print("runs/detect/predict/")
    print("====================================")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    main()