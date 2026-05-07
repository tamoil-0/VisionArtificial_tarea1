"""
Practica 02 - Introduccion a OpenCV y lectura de imagenes
=========================================================

Contenido:
    1.  Lectura de imagenes (color y escala de grises)
    2.  Conversion entre espacios de color (BGR, GRAY, HSV)
    3.  Separacion de canales HSV
    4.  Trabajo con pixeles (lectura, modificacion, slicing, sustitucion de zonas)
    5.  Transformaciones geometricas (traslacion, rotacion, escalado, perspectiva)
    6.  Operaciones aritmeticas (brillo, oscurecimiento)
    7.  Deteccion de movimiento (frame a frame y bucle completo)
    8.  Binarizacion y umbralizaciones (binaria, invertida, truncada, OTSU, adaptativa)

Uso:
    python practica2.py

Requisitos:
    pip install opencv-python numpy matplotlib
"""

import os
import shutil
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuracion: rutas a los archivos disponibles en la carpeta
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_IMAGEN = os.path.join(BASE_DIR, "imagen.jpg")
RUTA_VIDEO = os.path.join(BASE_DIR, "personas.mp4")


# ---------------------------------------------------------------------------
# I/O unicode-safe (la ruta tiene "°", que rompe cv2.imread/VideoCapture
# en Windows porque usan la API ANSI). np.fromfile soporta unicode.
# ---------------------------------------------------------------------------
def imread_u(path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se pudo abrir {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"No se pudo decodificar {path}")
    return img


def video_capture_u(path):
    """Devuelve (VideoCapture, ruta_temporal_o_None).

    Si VideoCapture no puede abrir la ruta original (porque tiene caracteres
    no-ASCII), copia el archivo a TEMP con un nombre ASCII y abre desde ahi.
    """
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, None
    cap.release()
    if not os.path.exists(path):
        return None, None
    ext = os.path.splitext(path)[1] or ".mp4"
    tmp = os.path.join(tempfile.gettempdir(), f"_cv_video_practica{ext}")
    shutil.copy(path, tmp)
    cap = cv2.VideoCapture(tmp)
    return (cap if cap.isOpened() else None), tmp


# ---------------------------------------------------------------------------
# Utilidades para mostrar imagenes (reemplazo de cv2_imshow de Colab)
# ---------------------------------------------------------------------------
def mostrar(img, titulo="Imagen", cmap=None):
    """Muestra una imagen con matplotlib. Convierte BGR -> RGB si es a color."""
    plt.figure(figsize=(7, 5))
    if cmap is None and img.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap=cmap or "gray")
    plt.title(titulo)
    plt.axis("off")
    plt.show()


def mostrar_varias(imagenes, titulos, cmap=None):
    """Muestra varias imagenes en una sola figura."""
    n = len(imagenes)
    plt.figure(figsize=(5 * n, 5))
    for i, (im, t) in enumerate(zip(imagenes, titulos), start=1):
        plt.subplot(1, n, i)
        if cmap is None and im.ndim == 3:
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(im, cmap=cmap or "gray")
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 1. Lectura de imagenes
# ===========================================================================
def leer_imagenes():
    """Lee la imagen en color y en escala de grises e imprime sus dimensiones."""
    img = imread_u(RUTA_IMAGEN)
    print(f"[1] Forma de la imagen a color (alto, ancho, canales): {img.shape}")

    gris = imread_u(RUTA_IMAGEN, cv2.IMREAD_GRAYSCALE)
    print(f"[1] Forma en escala de grises (alto, ancho): {gris.shape}")

    mostrar_varias([img, gris], ["Original (BGR)", "Escala de grises"])
    return img, gris


# ===========================================================================
# 2. Conversion entre espacios de color
# ===========================================================================
def conversion_espacios_color(img):
    """Convierte BGR -> GRAY y BGR -> HSV y compara las tres."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mostrar_varias([img, gris, hsv], ["Original", "Grises", "HSV (visual)"])
    return hsv


# ===========================================================================
# 3. Separacion de canales HSV
# ===========================================================================
def separar_canales_hsv(hsv):
    """Separa los tres canales del espacio HSV."""
    h, s, v = cv2.split(hsv)
    mostrar_varias([h, s, v], ["H (matiz)", "S (saturacion)", "V (valor)"])


# ===========================================================================
# 4. Trabajo con pixeles
# ===========================================================================
def trabajo_pixeles(img):
    """Lee, modifica y dibuja un punto rojo manipulando pixeles."""
    pixel = img[100, 150]
    print(f"[4] Pixel en (100,150) = {pixel}  (orden BGR)")

    # Actividad: dibujar un pequeno punto rojo modificando varios pixeles.
    img_punto = img.copy()
    img_punto[95:106, 145:156] = [0, 0, 255]  # cuadrado 11x11 rojo
    mostrar(img_punto, "Punto rojo dibujado en (100,150)")

    # Slicing: imprimir una pequena ventana de pixeles
    print("[4] Slicing img[0:5, 0:5] =")
    print(img[0:5, 0:5])

    # Extraer una zona (rostro/objeto si la imagen lo permite)
    zona = img[100:300, 100:300]
    mostrar(zona, "Zona extraida [100:300, 100:300]")

    # Sustitucion de una zona por otra
    img2 = img.copy()
    alto, ancho = img2.shape[:2]
    if alto >= 300 and ancho >= 300:
        img2[50:150, 50:150] = img[200:300, 200:300]
        mostrar(img2, "Sustitucion de zonas")


# ===========================================================================
# 5. Transformaciones geometricas
# ===========================================================================
def transformaciones_geometricas(img):
    filas, columnas = img.shape[:2]

    # Traslacion (100 px a la derecha, 50 px abajo)
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    trasladada = cv2.warpAffine(img, M, (columnas, filas))
    mostrar(trasladada, "Traslacion (+100, +50)")

    # Rotacion 45 grados respecto al centro
    centro = (columnas // 2, filas // 2)
    M_rot = cv2.getRotationMatrix2D(centro, 45, 1.0)
    rotada = cv2.warpAffine(img, M_rot, (columnas, filas))
    mostrar(rotada, "Rotacion 45 grados")

    # Actividad: rotar 90, 180 y 270 grados
    rotaciones = []
    for ang in (90, 180, 270):
        M_a = cv2.getRotationMatrix2D(centro, ang, 1.0)
        rotaciones.append(cv2.warpAffine(img, M_a, (columnas, filas)))
    mostrar_varias(rotaciones, ["90 grados", "180 grados", "270 grados"])

    # Escalado
    pequena = cv2.resize(img, None, fx=0.5, fy=0.5)
    grande = cv2.resize(img, None, fx=2, fy=2)
    print(f"[5] Tamano original : {img.shape[:2]}")
    print(f"[5] Tamano pequeno  : {pequena.shape[:2]}")
    print(f"[5] Tamano grande   : {grande.shape[:2]}")
    mostrar(pequena, "Escalado x0.5")
    mostrar(grande, "Escalado x2")

    # Transformacion de perspectiva (simula otra angulacion de camara)
    pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250], [250, 200]])
    M_p = cv2.getPerspectiveTransform(pts1, pts2)
    perspectiva = cv2.warpPerspective(img, M_p, (columnas, filas))
    mostrar(perspectiva, "Transformacion de perspectiva")


# ===========================================================================
# 6. Operaciones aritmeticas
# ===========================================================================
def operaciones_aritmeticas(img):
    """Suma y resta un valor constante para subir/bajar el brillo."""
    brillo = cv2.add(img, 50)
    oscura = cv2.subtract(img, 50)
    mostrar_varias([oscura, img, brillo], ["Oscurecida (-50)", "Original", "Brillo (+50)"])


# ===========================================================================
# 7. Deteccion de movimiento
# ===========================================================================
def deteccion_movimiento_basica():
    """Compara dos frames consecutivos del video con cv2.absdiff."""
    if not os.path.exists(RUTA_VIDEO):
        print("[7] No se encontro personas.mp4, se omite la deteccion de movimiento.")
        return

    video, _tmp = video_capture_u(RUTA_VIDEO)
    if video is None:
        print("[7] No se pudo abrir el video.")
        return
    ret1, frame1 = video.read()
    ret2, frame2 = video.read()
    video.release()
    if not (ret1 and ret2):
        print("[7] No se pudieron leer dos frames del video.")
        return

    diferencia = cv2.absdiff(frame1, frame2)
    mostrar_varias(
        [frame1, frame2, diferencia],
        ["Frame 1", "Frame 2", "absdiff(frame1, frame2)"],
    )


def deteccion_movimiento_completa(max_frames=60):
    """Bucle de deteccion de movimiento con umbralizacion binaria.

    Procesa hasta max_frames pares de frames para no quedarse colgado, y al
    final muestra el ultimo mapa de movimiento detectado.
    """
    if not os.path.exists(RUTA_VIDEO):
        print("[7] No se encontro personas.mp4, se omite la deteccion completa.")
        return

    video, _tmp = video_capture_u(RUTA_VIDEO)
    if video is None:
        print("[7] No se pudo abrir el video.")
        return
    ultimo_th = None
    contador = 0
    while video.isOpened() and contador < max_frames:
        ret, frame1 = video.read()
        ret2, frame2 = video.read()
        if not (ret and ret2):
            break
        diff = cv2.absdiff(frame1, frame2)
        gris = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gris, 25, 255, cv2.THRESH_BINARY)
        ultimo_th = th
        contador += 1
    video.release()

    if ultimo_th is not None:
        mostrar(ultimo_th, f"Movimiento detectado (ultimo de {contador} pares)")


# ===========================================================================
# 8. Binarizacion / umbralizacion
# ===========================================================================
def binarizacion(gris):
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    _, invertida = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)
    _, trunc = cv2.threshold(gris, 127, 255, cv2.THRESH_TRUNC)
    _, cero = cv2.threshold(gris, 127, 255, cv2.THRESH_TOZERO)
    _, cero_inv = cv2.threshold(gris, 127, 255, cv2.THRESH_TOZERO_INV)

    mostrar_varias(
        [binaria, invertida, trunc, cero, cero_inv],
        ["BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"],
    )

    # Otsu: el umbral se calcula automaticamente
    umbral_otsu, otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"[8] Umbral OTSU calculado automaticamente: {umbral_otsu:.1f}")

    # Adaptativa (umbral variable por region)
    adapt = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    mostrar_varias([gris, otsu, adapt], ["Original gris", "OTSU", "Adaptativa Gauss"])


# ===========================================================================
# Programa principal
# ===========================================================================
def main():
    print("=" * 60)
    print("PRACTICA 02 - Introduccion a OpenCV y lectura de imagenes")
    print("=" * 60)

    img, gris = leer_imagenes()
    hsv = conversion_espacios_color(img)
    separar_canales_hsv(hsv)
    trabajo_pixeles(img)
    transformaciones_geometricas(img)
    operaciones_aritmeticas(img)
    deteccion_movimiento_basica()
    deteccion_movimiento_completa()
    binarizacion(gris)

    print("\n[OK] Practica 2 finalizada.")


if __name__ == "__main__":
    main()
