FROM python:3.9-slim

# Installer les dépendances système (zbar pour pyzbar + libgl1 pour OpenCV)
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Créer l'utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copier les requirements d'abord pour mieux utiliser le cache Docker
COPY --chown=app:app requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY --chown=app:app . .

# Exposer le port
EXPOSE 8000

# Lancer l'application avec optimisations
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]