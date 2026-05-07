"""
Practica 04 - Filtrado, Morfologia y Deteccion de Bordes
=========================================================

Objetivo:
    Aplicar tecnicas de filtrado y morfologia para reducir ruido, mejorar
    imagenes y detectar bordes con OpenCV.

Contenido:
    1.  Filtros pasa baja con distintos kernels (3x3, 7x7, 15x15)
    2.  Blurring (promedio)
    3.  Desenfoque Gaussiano
    4.  Filtro de mediana (utilidad: ruido sal y pimienta)
    5.  Filtro bilateral (suaviza pero conserva bordes)
    6.  "Restauracion" de imagen antigua (mediana + ecualizacion)
    7.  Transformaciones morfologicas (erosion, dilatacion, apertura, cierre)
    8.  Filtro pasa alta - Sobel X / Y / magnitud
    9.  Filtro Canny y coloracion de bordes

Uso:
    python practica4.py

Requisitos:
    pip install opencv-python numpy matplotlib
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_IMAGEN = os.path.join(BASE_DIR, "imagen.jpg")
RUTA_IMAGEN_2 = os.path.join(BASE_DIR, "imagen2.jpg")  # se usa como "antigua"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def imread_u(path, flags=cv2.IMREAD_COLOR):
    """imread compatible con rutas que tienen caracteres no-ASCII (Windows)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se pudo abrir {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"No se pudo decodificar {path}")
    return img


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def cargar_color():
    img = imread_u(RUTA_IMAGEN)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ya en RGB para matplotlib


def cargar_gris():
    return imread_u(RUTA_IMAGEN, cv2.IMREAD_GRAYSCALE)


# ===========================================================================
# 1. Filtros pasa baja - efecto del tamano del kernel
# ===========================================================================
def filtros_pasa_baja(img_rgb):
    """A mayor kernel, mayor suavizado y mas detalle perdido."""
    k3 = cv2.blur(img_rgb, (3, 3))
    k7 = cv2.blur(img_rgb, (7, 7))
    k15 = cv2.blur(img_rgb, (15, 15))

    plt.figure(figsize=(16, 4))
    for i, (im, t) in enumerate(
        zip([img_rgb, k3, k7, k15], ["Original", "(3,3)", "(7,7)", "(15,15)"]),
        start=1,
    ):
        plt.subplot(1, 4, i)
        plt.imshow(im)
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 2-3-4. Blur / Gaussian / Mediana
# ===========================================================================
def comparar_suavizados(img_rgb):
    blur = cv2.blur(img_rgb, (9, 9))
    gauss = cv2.GaussianBlur(img_rgb, (9, 9), 0)
    median = cv2.medianBlur(img_rgb, 9)

    plt.figure(figsize=(16, 4))
    for i, (im, t) in enumerate(
        zip(
            [img_rgb, blur, gauss, median],
            ["Original", "Blur (9x9)", "Gaussian (9x9)", "Median (9)"],
        ),
        start=1,
    ):
        plt.subplot(1, 4, i)
        plt.imshow(im)
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def demo_mediana_sal_pimienta(img_rgb):
    """Anade ruido sal y pimienta a la imagen y lo limpia con cv2.medianBlur."""
    ruidosa = img_rgb.copy()
    rng = np.random.default_rng(0)
    cantidad = 0.02
    n = int(cantidad * ruidosa.shape[0] * ruidosa.shape[1])

    # Sal (blancos)
    ys = rng.integers(0, ruidosa.shape[0], n)
    xs = rng.integers(0, ruidosa.shape[1], n)
    ruidosa[ys, xs] = 255
    # Pimienta (negros)
    ys = rng.integers(0, ruidosa.shape[0], n)
    xs = rng.integers(0, ruidosa.shape[1], n)
    ruidosa[ys, xs] = 0

    limpia = cv2.medianBlur(ruidosa, 5)

    plt.figure(figsize=(15, 4))
    for i, (im, t) in enumerate(
        zip([img_rgb, ruidosa, limpia], ["Original", "Con ruido", "Median Blur 5"]),
        start=1,
    ):
        plt.subplot(1, 3, i)
        plt.imshow(im)
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 5. Filtro bilateral
# ===========================================================================
def filtro_bilateral(img_rgb):
    """Suaviza zonas planas pero conserva los bordes. Util en retoque facial."""
    bilateral = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    gauss = cv2.GaussianBlur(img_rgb, (9, 9), 0)
    plt.figure(figsize=(15, 4))
    for i, (im, t) in enumerate(
        zip([img_rgb, gauss, bilateral], ["Original", "Gaussian", "Bilateral"]),
        start=1,
    ):
        plt.subplot(1, 3, i)
        plt.imshow(im)
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 6. Restauracion de imagen "antigua"
# ===========================================================================
def restauracion_antigua():
    """Aplica mediana + ecualizacion para mejorar una foto en grises."""
    ruta = RUTA_IMAGEN_2 if os.path.exists(RUTA_IMAGEN_2) else RUTA_IMAGEN
    img = imread_u(ruta, cv2.IMREAD_GRAYSCALE)
    restaurada = cv2.medianBlur(img, 5)
    restaurada = cv2.equalizeHist(restaurada)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Antigua: {os.path.basename(ruta)}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(restaurada, cmap="gray")
    plt.title("Restaurada (mediana + ecualizacion)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 7. Morfologia - se construye una imagen binaria sintetica
# ===========================================================================
def crear_imagen_binaria_de_prueba():
    """Genera una imagen binaria con figuras y ruido para practicar morfologia."""
    img = np.zeros((300, 500), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
    cv2.circle(img, (350, 120), 60, 255, -1)
    cv2.rectangle(img, (250, 220), (450, 270), 255, -1)
    # Ruido tipo "sal" para que la apertura tenga algo que limpiar
    rng = np.random.default_rng(42)
    for _ in range(400):
        y = rng.integers(0, img.shape[0])
        x = rng.integers(0, img.shape[1])
        img[y, x] = 255
    # Y unos huecos para que el cierre tenga algo que rellenar
    for _ in range(60):
        y = rng.integers(60, 190)
        x = rng.integers(60, 190)
        img[y, x] = 0
    return img


def transformaciones_morfologicas():
    img = crear_imagen_binaria_de_prueba()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    erosion = cv2.erode(img, kernel, iterations=1)
    dilatacion = cv2.dilate(img, kernel, iterations=1)
    apertura = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)   # erosion + dilatacion
    cierre = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)   # dilatacion + erosion

    plt.figure(figsize=(15, 8))
    for i, (im, t) in enumerate(
        zip(
            [img, erosion, dilatacion, apertura, cierre],
            [
                "Binaria con ruido",
                "Erosion (achica blancos)",
                "Dilatacion (engorda blancos)",
                "Apertura (limpia puntos blancos)",
                "Cierre (rellena huecos negros)",
            ],
        ),
        start=1,
    ):
        plt.subplot(2, 3, i)
        plt.imshow(im, cmap="gray")
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 8. Filtro Sobel
# ===========================================================================
def filtro_sobel(gris):
    """Sobel X = derivada horizontal -> detecta bordes verticales.
    Sobel Y = derivada vertical   -> detecta bordes horizontales.
    """
    sobelx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = cv2.magnitude(sobelx, sobely)

    plt.figure(figsize=(15, 4))
    for i, (im, t) in enumerate(
        zip(
            [gris, np.abs(sobelx), np.abs(sobely), magnitud],
            ["Original", "|Sobel X|", "|Sobel Y|", "Magnitud Sobel"],
        ),
        start=1,
    ):
        plt.subplot(1, 4, i)
        plt.imshow(im, cmap="gray")
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 9. Canny y coloracion de bordes
# ===========================================================================
def filtro_canny():
    img_color = imread_u(RUTA_IMAGEN)
    gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Tres pares de umbrales para mostrar el efecto
    canny1 = cv2.Canny(gris, 50, 150)
    canny2 = cv2.Canny(gris, 100, 200)
    canny3 = cv2.Canny(gris, 150, 250)

    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB).copy()
    img_rgb[canny2 > 0] = [255, 0, 0]  # bordes en rojo sobre la imagen original

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(canny1, cmap="gray")
    plt.title("Canny (50, 150)")
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(canny2, cmap="gray")
    plt.title("Canny (100, 200)")
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(canny3, cmap="gray")
    plt.title("Canny (150, 250)")
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(img_rgb)
    plt.title("Bordes coloreados (rojo)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Programa principal
# ===========================================================================
def main():
    print("=" * 60)
    print("PRACTICA 04 - Filtrado, morfologia y deteccion de bordes")
    print("=" * 60)

    img_rgb = cargar_color()
    gris = cargar_gris()

    filtros_pasa_baja(img_rgb)
    comparar_suavizados(img_rgb)
    demo_mediana_sal_pimienta(img_rgb)
    filtro_bilateral(img_rgb)
    restauracion_antigua()
    transformaciones_morfologicas()
    filtro_sobel(gris)
    filtro_canny()

    print("\n[OK] Practica 4 finalizada.")


if __name__ == "__main__":
    main()
