import cv2
import os
import tempfile
from pyzbar.pyzbar import decode
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Barcode Reader API", 
    description="API rapide pour lire les codes-barres à partir d'images", 
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pool de threads pour les opérations CPU-bound
thread_pool = ThreadPoolExecutor(max_workers=4)

# Cache pour les images prétraitées (optionnel, pour requêtes répétées)
image_cache = {}
CACHE_MAX_SIZE = 100

def preprocess_image_fast(image_data: np.ndarray) -> np.ndarray:
    """
    Prétraitement optimisé de l'image en mémoire sans écriture disque.
    """
    # Conversion en niveaux de gris
    if len(image_data.shape) == 3:
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_data
    
    # Filtrage adaptatif - plus rapide que bilateralFilter
    gray = cv2.medianBlur(gray, 3)
    
    # Égalisation d'histogramme pour améliorer le contraste
    gray = cv2.equalizeHist(gray)
    
    return gray

def preprocess_image_quality(image_data: np.ndarray) -> np.ndarray:
    """
    Version avec meilleure qualité mais un peu plus lente.
    À utiliser si la version fast ne donne pas de bons résultats.
    """
    if len(image_data.shape) == 3:
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_data
    
    # Filtrage préservant les bords (meilleur qualité)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # Réduction du bruit
    gray = cv2.fastNlMeansDenoising(gray)
    
    return gray

async def read_barcodes_optimized(image_data: np.ndarray, use_quality_preprocess: bool = False) -> list:
    """
    Lecture des codes-barres avec prétraitement optimisé.
    """
    try:
        # Choix de la méthode de prétraitement
        if use_quality_preprocess:
            processed_image = preprocess_image_quality(image_data)
        else:
            processed_image = preprocess_image_fast(image_data)
        
        # Décodage des codes-barres dans un thread séparé
        loop = asyncio.get_event_loop()
        barcodes = await loop.run_in_executor(
            thread_pool, 
            decode, 
            processed_image
        )
        
        results = []
        for barcode in barcodes:
            try:
                data = barcode.data.decode('utf-8')
                results.append({
                    "type": barcode.type,
                    "data": data,
                    "quality": barcode.quality if hasattr(barcode, 'quality') else None
                })
            except UnicodeDecodeError:
                # Fallback pour l'encodage
                try:
                    data = barcode.data.decode('latin-1')
                    results.append({
                        "type": barcode.type,
                        "data": data,
                        "quality": barcode.quality if hasattr(barcode, 'quality') else None
                    })
                except:
                    logger.warning(f"Impossible de décoder le code-barres de type {barcode.type}")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des codes-barres: {str(e)}")
        raise

def decode_image_file(file_content: bytes) -> np.ndarray:
    """
    Décode le contenu du fichier en image numpy array.
    """
    try:
        # Conversion directe des bytes en image numpy
        image_array = np.frombuffer(file_content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Impossible de décoder l'image")
        return image
    except Exception as e:
        logger.error(f"Erreur décodage image: {str(e)}")
        raise

@app.post("/scan-barcode")
async def scan_barcode(
    file: UploadFile = File(...),
    quality_mode: bool = False,
    max_file_size: int = 10 * 1024 * 1024  # 10MB max
):
    """
    Endpoint optimisé pour la lecture de codes-barres.
    
    Args:
        quality_mode: Si True, utilise un prétraitement de meilleure qualité (plus lent)
        max_file_size: Taille maximale du fichier en bytes
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Vérification du type de fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="Le fichier doit être une image"
            )

        # Lecture du fichier avec limite de taille
        file_content = await file.read()
        if len(file_content) > max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"Fichier trop volumineux. Maximum: {max_file_size} bytes"
            )

        # Décodage de l'image
        image_data = decode_image_file(file_content)
        
        # Lecture des codes-barres
        codes = await read_barcodes_optimized(image_data, quality_mode)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Traitement terminé en {processing_time:.2f}s - {len(codes)} codes trouvés")

        if not codes:
            return JSONResponse({
                "success": False, 
                "message": "Aucun code-barres détecté.",
                "processing_time": f"{processing_time:.2f}s"
            }, status_code=404)

        return {
            "success": True, 
            "count": len(codes), 
            "barcodes": codes,
            "processing_time": f"{processing_time:.2f}s",
            "quality_mode": quality_mode
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du scan: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors du traitement de l'image: {str(e)}"
        )

@app.post("/scan-barcode-batch")
async def scan_barcode_batch(files: list[UploadFile] = File(...)):
    """
    Endpoint pour traiter plusieurs images en parallèle.
    """
    try:
        tasks = []
        for file in files:
            task = scan_barcode(file)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_scans = []
        failed_scans = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_scans.append({
                    "filename": files[i].filename,
                    "error": str(result)
                })
            else:
                successful_scans.append({
                    "filename": files[i].filename,
                    "result": result
                })
        
        return {
            "success": True,
            "total_files": len(files),
            "successful_scans": successful_scans,
            "failed_scans": failed_scans
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint de santé de l'API."""
    return {
        "status": "healthy",
        "version": "2.0",
        "optimized": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Pour éviter les problèmes de concurrence avec pyzbar
        loop="asyncio"
    )