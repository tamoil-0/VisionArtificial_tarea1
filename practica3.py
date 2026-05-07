"""
Practica 03 - Histogramas y Mejora de Contraste
================================================

Objetivo:
    Analizar la distribucion de intensidades de una imagen mediante histogramas
    y aplicar tecnicas de mejora de contraste con OpenCV.

Contenido:
    1.  Histograma en escala de grises
    2.  Histograma de imagen a color (canales B, G, R)
    3.  Histograma con mascara (region de interes)
    4.  Ecualizacion del histograma
    5.  Comparacion de histogramas original vs ecualizada
    6.  Ecualizacion adaptativa (CLAHE)

Uso:
    python practica3.py

Requisitos:
    pip install opencv-python numpy matplotlib
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_IMAGEN = os.path.join(BASE_DIR, "imagen.jpg")
RUTA_IMAGEN_2 = os.path.join(BASE_DIR, "imagen2.jpg")  # opcional para comparar


# ---------------------------------------------------------------------------
# Loader unicode-safe (la ruta tiene "°", que rompe cv2.imread en Windows).
# ---------------------------------------------------------------------------
def imread_u(path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se pudo abrir {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"No se pudo decodificar {path}")
    return img


# ===========================================================================
# 1. Histograma en escala de grises
# ===========================================================================
def histograma_grises(ruta):
    """Calcula y dibuja el histograma de una imagen en escala de grises.

    Interpretacion:
        - Concentrado a la izquierda  -> imagen oscura
        - Concentrado a la derecha    -> imagen clara
        - Extendido sobre todo el eje -> buen contraste
    """
    img = imread_u(ruta, cv2.IMREAD_GRAYSCALE)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Imagen: {os.path.basename(ruta)}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(hist, color="black")
    plt.title("Histograma en escala de grises")
    plt.xlabel("Nivel de intensidad (0-255)")
    plt.ylabel("Cantidad de pixeles")
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()
    return img


# ===========================================================================
# 2. Histograma de imagen a color (un trazo por canal)
# ===========================================================================
def histograma_color(ruta):
    """Histograma por canal: azul = 0, verde = 1, rojo = 2."""
    img = imread_u(ruta)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Imagen a color")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    for i, color in enumerate(("b", "g", "r")):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=f"Canal {color.upper()}")
    plt.title("Histograma RGB")
    plt.xlabel("Nivel de intensidad")
    plt.ylabel("Cantidad de pixeles")
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 3. Histograma con mascara (region de interes)
# ===========================================================================
def histograma_con_mascara(ruta):
    """Calcula el histograma solamente de una region rectangular de la imagen.

    En la mascara: pixeles blancos = se cuentan, pixeles negros = se ignoran.
    """
    img = imread_u(ruta, cv2.IMREAD_GRAYSCALE)

    alto, ancho = img.shape
    mascara = np.zeros_like(img)
    # Region central robusta (no depende del tamano exacto de la imagen)
    y1, y2 = alto // 4, 3 * alto // 4
    x1, x2 = ancho // 4, 3 * ancho // 4
    mascara[y1:y2, x1:x2] = 255

    hist_total = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_region = cv2.calcHist([img], [0], mascara, [256], [0, 256])

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mascara, cmap="gray")
    plt.title("Mascara (zona analizada)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(hist_total, color="gray", label="Toda la imagen")
    plt.plot(hist_region, color="red", label="Solo la region")
    plt.title("Comparacion de histogramas")
    plt.xlabel("Nivel de intensidad")
    plt.ylabel("Cantidad de pixeles")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 4. Ecualizacion del histograma
# ===========================================================================
def ecualizacion_histograma(ruta):
    """cv2.equalizeHist redistribuye las intensidades para mejorar el contraste."""
    img = imread_u(ruta, cv2.IMREAD_GRAYSCALE)
    ecualizada = cv2.equalizeHist(img)

    plt.figure(figsize=(12, 5))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(ecualizada, cmap="gray")
    plt.title("Ecualizada")
    plt.axis("off")

    hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([ecualizada], [0], None, [256], [0, 256])
    plt.subplot(2, 2, 3)
    plt.plot(hist1, color="black")
    plt.title("Histograma original")
    plt.xlim([0, 256])
    plt.subplot(2, 2, 4)
    plt.plot(hist2, color="black")
    plt.title("Histograma ecualizado")
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()
    return img, ecualizada


# ===========================================================================
# 5. Comparar superpuestos los histogramas
# ===========================================================================
def comparar_histogramas(img, ecualizada):
    hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([ecualizada], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 4))
    plt.plot(hist1, label="Original", color="blue")
    plt.plot(hist2, label="Ecualizada", color="red")
    plt.title("Histogramas: original vs ecualizada")
    plt.xlabel("Nivel de intensidad")
    plt.ylabel("Cantidad de pixeles")
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 6. CLAHE - Ecualizacion adaptativa con limite de contraste
# ===========================================================================
def ecualizacion_adaptativa(ruta):
    """CLAHE evita que la ecualizacion exagere demasiado algunas zonas.

    Trabaja por bloques (tileGridSize) y limita el contraste por bloque
    (clipLimit) para no amplificar el ruido.
    """
    img = imread_u(ruta, cv2.IMREAD_GRAYSCALE)
    ecualizada = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptativa = clahe.apply(img)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(ecualizada, cmap="gray")
    plt.title("Ecualizacion normal")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(adaptativa, cmap="gray")
    plt.title("CLAHE (adaptativa)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Programa principal
# ===========================================================================
def main():
    print("=" * 60)
    print("PRACTICA 03 - Histogramas y mejora de contraste")
    print("=" * 60)

    histograma_grises(RUTA_IMAGEN)

    # Comparacion entre dos imagenes (clara vs oscura, si existe la segunda)
    if os.path.exists(RUTA_IMAGEN_2):
        print("[1] Comparando imagen.jpg vs imagen2.jpg...")
        histograma_grises(RUTA_IMAGEN_2)

    histograma_color(RUTA_IMAGEN)
    histograma_con_mascara(RUTA_IMAGEN)
    img, ecu = ecualizacion_histograma(RUTA_IMAGEN)
    comparar_histogramas(img, ecu)
    ecualizacion_adaptativa(RUTA_IMAGEN)

    print("\n[OK] Practica 3 finalizada.")


if __name__ == "__main__":
    main()
