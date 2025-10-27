import cv2
import numpy as np
from pyzbar.pyzbar import decode
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

app = FastAPI(
    title="Barcode Reader API",
    description="API rapide pour lire les codes-barres à partir d'images",
    version="1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image_bytes(image_bytes: bytes):
    """Prétraitement rapide sans I/O disque."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Impossible de décoder l'image")

    # Convertir en niveaux de gris et lisser légèrement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filtre bilatéral remplacé par GaussianBlur (plus rapide, précision équivalente pour les barcodes)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def read_barcodes_from_image(image: np.ndarray):
    """Lecture directe sans sauvegarde de fichier."""
    barcodes = decode(image)
    results = []
    for barcode in barcodes:
        results.append({
            "type": barcode.type,
            "data": barcode.data.decode('utf-8')
        })
    return results


@app.post("/scan-barcode")
async def scan_barcode(file: UploadFile = File(...)):
    """Endpoint principal : reçoit une image et retourne les codes-barres trouvés."""
    try:
        # Lecture en mémoire sans écriture sur disque
        file_bytes = await file.read()

        # Prétraitement léger et rapide
        processed_img = preprocess_image_bytes(file_bytes)

        # Lecture du code-barres
        barcodes = read_barcodes_from_image(processed_img)

        if not barcodes:
            return JSONResponse(
                {"success": False, "message": "Aucun code-barres détecté."},
                status_code=404
            )

        return {"success": True, "count": len(barcodes), "barcodes": barcodes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
