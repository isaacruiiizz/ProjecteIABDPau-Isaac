#!/bin/bash

# Script d'instal·lació de Label Studio per anotacions YOLO
# Autor: Francesc Barragán
# Data: Novembre 2025

echo "=== Instal·lació de Label Studio ==="

# Crear entorn virtual
python3 -m venv label_studio_env
source label_studio_env/bin/activate

# Instal·lar Label Studio
pip install --upgrade pip
pip install label-studio

# Instal·lar dependències addicionals per treballar amb imatges
pip install pillow opencv-python

echo "=== Instal·lació completada ==="
echo ""
echo "Per iniciar Label Studio executa:"
echo "  source label_studio_env/bin/activate"
echo "  label-studio start"
echo ""
echo "Després obre el navegador a: http://localhost:8080"
