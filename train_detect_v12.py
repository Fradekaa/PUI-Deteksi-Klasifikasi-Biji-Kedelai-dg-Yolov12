# ==========================================
# TRAIN YOLOv12 DETECTION
# SoyGuard AI
# ==========================================

from ultralytics import YOLO

model = YOLO("yolo12s.pt")

def main():
    # ======================================
    # START TRAINING
    # ======================================

    model.train(

        # ==================================
        # DATASET
        # ==================================

        data=r"D:\Kuliah\Semester 6\PUI\soybean cls\data.yaml",

        # ==================================
        # TRAINING PARAMETER
        # ==================================

        epochs=100,

        imgsz = 640,
        
        batch = 8,

        device = 0,

        workers=0,

        pretrained=True,

        # ==================================
        # OPTIMIZER
        # ==================================

        optimizer="AdamW",

        lr0=0.0005,

        weight_decay=0.0005,

        momentum=0.937,

        # ==================================
        # AUGMENTATION
        # ==================================

        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5,

        translate=0.10,

        scale=0.50,

        shear=0,

        perspective=0,

        flipud=0,

        fliplr=0.50,

        mosaic=1.0,

        mixup=0.10,

        copy_paste=0,

        # ==================================
        # EARLY STOPPING
        # ==================================

        patience=20,

        # ==================================
        # SAVE RESULT
        # ==================================

        project="runs",

        name="soybean_detect_v12",

        exist_ok=True,

        save=True,

        save_period=-1,

        plots=True,

        verbose=True

    )

    print("\n========================================")
    print("Training selesai.")
    print("Model terbaik tersimpan pada:")
    print("runs/detect/soybean_detect_v12/weights/best.pt")
    print("========================================")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    main()