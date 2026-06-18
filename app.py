# ==========================================================
# SoyGuard AI
# app.py
# YOLOv12 Detection Only
# ==========================================================

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    url_for
)

from ultralytics import YOLO

import cv2
import os
import shutil

from reportlab.pdfgen import canvas

# ==========================================================
# FLASK
# ==========================================================

app = Flask(__name__)

# ==========================================================
# FOLDER
# ==========================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

CROP_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "crops"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "reports"
)

for folder in [

    UPLOAD_FOLDER,

    RESULT_FOLDER,

    CROP_FOLDER,

    REPORT_FOLDER

]:

    os.makedirs(

        folder,

        exist_ok=True

    )

# ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "detect.pt"
)

model = YOLO(MODEL_PATH)

# ==========================================================
# GOOD CLASS
# ==========================================================

GOOD_CLASSES = [

    "intact_soybeans"

]

# ==========================================================
# DEFECT CLASS
# ==========================================================

DEFECT_CLASSES = [

    "broken_soybeans",

    "skin_damaged_soybeans",

    "immature_soybeans"

]

# ==========================================================
# CLEAR FOLDER
# ==========================================================

def clear_folder(folder):

    if not os.path.exists(folder):

        return

    for file in os.listdir(folder):

        path = os.path.join(

            folder,

            file

        )

        try:

            if os.path.isfile(path):

                os.remove(path)

            else:

                shutil.rmtree(path)

        except:

            pass

# ==========================================================
# GRADE
# ==========================================================

def grading(score):

    if score >= 90:

        return "A", "Excellent"

    elif score >= 75:

        return "B", "Good"

    elif score >= 60:

        return "C", "Fair"

    else:

        return "D", "Poor"

# ==========================================================
# PDF REPORT
# ==========================================================

def generate_pdf(

    total_soybean,

    total_intact,

    total_defect,

    quality_score,

    grade,

    quality

):

    pdf_path = os.path.join(

        REPORT_FOLDER,

        "SoyGuard_Report.pdf"

    )

    pdf = canvas.Canvas(pdf_path)

    pdf.setFont(

        "Helvetica-Bold",

        18

    )

    pdf.drawString(

        50,

        800,

        "SoyGuard AI Report"

    )

    pdf.setFont(

        "Helvetica",

        12

    )

    pdf.drawString(

        50,

        760,

        f"Total Soybean : {total_soybean}"

    )

    pdf.drawString(

        50,

        735,

        f"Total Intact : {total_intact}"

    )

    pdf.drawString(

        50,

        710,

        f"Total Defect : {total_defect}"

    )

    pdf.drawString(

        50,

        685,

        f"Quality Score : {quality_score:.2f}%"

    )

    pdf.drawString(

        50,

        660,

        f"Grade : {grade}"

    )

    pdf.drawString(

        50,

        635,

        f"Quality : {quality}"

    )

    pdf.save()

    return pdf_path

# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# PREDICT
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # ======================================
    # CLEAR OLD FILE
    # ======================================

    clear_folder(UPLOAD_FOLDER)
    clear_folder(RESULT_FOLDER)
    clear_folder(CROP_FOLDER)

    # ======================================
    # CHECK FILE
    # ======================================

    if "images" not in request.files:

        return render_template(
            "index.html"
        )

    file = request.files["images"]

    if file.filename == "":

        return render_template(
            "index.html"
        )

    # ======================================
    # SAVE IMAGE
    # ======================================

    filename = file.filename

    upload_path = os.path.join(

        UPLOAD_FOLDER,

        filename

    )

    file.save(upload_path)

    # ======================================
    # READ IMAGE
    # ======================================

    image = cv2.imread(upload_path)

    if image is None:

        return render_template(
            "index.html"
        )

    # ======================================
    # YOLO DETECTION
    # ======================================

    detect_result = model.predict(

        source=image,

        imgsz=960,

        conf=0.25,

        save=False,

        verbose=False

    )

    result = detect_result[0]

    # ======================================
    # DRAW BOUNDING BOX
    # ======================================

    plotted_image = result.plot()

    result_name = "result.jpg"

    result_path = os.path.join(

        RESULT_FOLDER,

        result_name

    )

    cv2.imwrite(

        result_path,

        plotted_image

    )

    # ======================================
    # VARIABLE
    # ======================================

    results = []

    statistics = {}

    total_soybean = 0

    total_intact = 0

    total_defect = 0

    quality_score = 0

    grade = "-"

    quality = "-"

    # ======================================
    # NEXT
    # Crop & Statistics
    # (Part 3)
    # ======================================
        # ======================================
    # LOOP DETECTION
    # ======================================

    for i, box in enumerate(result.boxes):

        total_soybean += 1

        # -------------------------------
        # Bounding Box
        # -------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # -------------------------------
        # Crop Image
        # -------------------------------

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:

            continue

        crop_name = f"crop_{i+1}.jpg"

        crop_path = os.path.join(

            CROP_FOLDER,

            crop_name

        )

        cv2.imwrite(

            crop_path,

            crop

        )

        # -------------------------------
        # Detection Class
        # -------------------------------

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        # -------------------------------
        # Statistics
        # -------------------------------

        statistics[class_name] = statistics.get(

            class_name,

            0

        ) + 1

        # -------------------------------
        # Intact / Defect
        # -------------------------------

        if class_name in GOOD_CLASSES:

            total_intact += 1

        else:

            total_defect += 1

        # -------------------------------
        # Save Result
        # -------------------------------

        results.append({

            "class_name": class_name.replace("_", " ").title(),

            "confidence": round(confidence * 100, 2),

            "crop_path": url_for(
                "static",
                filename=f"crops/{crop_name}"
            )

        })
        
    # ======================================
    # QUALITY SCORE
    # ======================================

    if total_soybean > 0:

        quality_score = round(

            (total_intact / total_soybean) * 100,

            2

        )

    else:

        quality_score = 0

    # ======================================
    # GRADE
    # ======================================

    grade, quality = grading(

        quality_score

    )

    # ======================================
    # GENERATE PDF
    # ======================================

    generate_pdf(

        total_soybean,

        total_intact,

        total_defect,

        quality_score,

        grade,

        quality

    )
    # ======================================
    # RENDER TEMPLATE
    # ======================================

    return render_template(

        "index.html",

        uploaded_image=url_for(
            "static",
            filename=f"uploads/{filename}"
        ),

        result_image=url_for(
            "static",
            filename=f"results/{result_name}"
        ),

        results=results,

        statistics=statistics,

        total_soybean=total_soybean,

        total_intact=total_intact,

        total_defect=total_defect,

        quality_score=quality_score,

        grade=grade,

        quality=quality

    )
# ==========================================================
# DOWNLOAD PDF
# ==========================================================

@app.route("/download")
def download():

    pdf_path = os.path.join(

        REPORT_FOLDER,

        "SoyGuard_Report.pdf"

    )

    if os.path.exists(pdf_path):

        return send_file(

            pdf_path,

            as_attachment=True

        )

    return "PDF Report tidak ditemukan."


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )