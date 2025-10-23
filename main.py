import cv2
import os
import tempfile
from PIL import Image
import zbarlight
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Barcode Reader API", description="API rapide pour lire les codes-barres à partir d'images", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_path: str):
    """Améliore la qualité de l'image pour optimiser la détection."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    temp_path = image_path.replace(".png", "_processed.png")
    cv2.imwrite(temp_path, gray)
    return temp_path

def read_barcodes(image_path: str):
    """Lit tous les codes-barres détectés dans une image."""
    processed_path = preprocess_image(image_path)
    
    with open(processed_path, 'rb') as image_file:
        image = Image.open(image_file)
        image.load()
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    codes = []
    try:
        barcodes = zbarlight.scan_codes(['ean13', 'ean8', 'upca', 'upce', 'code128', 'code39', 'qr'], image)
        if barcodes:
            for barcode in barcodes:
                codes.append({
                    "type": "detected",
                    "data": barcode.decode('utf-8')
                })
    except Exception as e:
        print(f"Error reading barcodes: {e}")
    
    os.remove(processed_path)
    return codes

@app.post("/scan-barcode")
async def scan_barcode(file: UploadFile = File(...)):
    """Endpoint principal : reçoit une image et retourne les codes-barres trouvés."""
    try:
        # Sauvegarde temporaire de l'image reçue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Lecture des codes-barres
        codes = read_barcodes(tmp_path)
        os.remove(tmp_path)

        if not codes:
            return JSONResponse({"success": False, "message": "Aucun code-barres détecté."}, status_code=404)

        return {"success": True, "count": len(codes), "barcodes": codes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Barcode Reader API is running!"}