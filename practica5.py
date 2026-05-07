"""
Practica 05 - Contornos y Aplicaciones
=======================================

Objetivo:
    Detectar, analizar y comparar contornos con OpenCV, y aplicar estos
    conceptos a problemas reales (conteo de objetos, videovigilancia).

Contenido:
    1.  Introduccion a contornos y conteo basico
    2.  Conteo de monedas (monedas.jpg)
    3.  Conteo de tornillos discriminando por area + morfologia
    4.  Centroide de cada contorno
    5.  Discriminacion por area (objetos grandes vs pequenos)
    6.  Prueba de polygon test (punto dentro/fuera de un contorno)
    7.  Comparacion de formas (engranaje1.jpg vs engranaje2.jpg)
    8.  Sistema de videovigilancia con HOG + zona segura (personas.mp4)

Uso:
    python practica5.py

Requisitos:
    pip install opencv-python numpy matplotlib
"""

import os
import shutil
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_IMAGEN = os.path.join(BASE_DIR, "imagen.jpg")
RUTA_MONEDAS = os.path.join(BASE_DIR, "monedas.jpg")
RUTA_TORNILLOS = os.path.join(BASE_DIR, "tornillos.jpg")
RUTA_ENGRANAJE_1 = os.path.join(BASE_DIR, "engranaje1.jpg")
RUTA_ENGRANAJE_2 = os.path.join(BASE_DIR, "engranaje2.jpg")
RUTA_VIDEO = os.path.join(BASE_DIR, "personas.mp4")


# ---------------------------------------------------------------------------
# I/O unicode-safe (cv2.imread y cv2.VideoCapture en Windows no soportan
# rutas con caracteres no-ASCII como "°"). np.fromfile si los soporta.
# ---------------------------------------------------------------------------
def imread_u(path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


def video_capture_u(path):
    """Devuelve VideoCapture o None. Si la ruta tiene unicode y falla,
    copia el archivo a TEMP con un nombre ASCII y abre desde ahi.
    """
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap
    cap.release()
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1] or ".mp4"
    tmp = os.path.join(tempfile.gettempdir(), f"_cv_video_practica5{ext}")
    shutil.copy(path, tmp)
    cap = cv2.VideoCapture(tmp)
    return cap if cap.isOpened() else None


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def mostrar_bgr(img_bgr, titulo):
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis("off")
    plt.show()


# ===========================================================================
# 1. Introduccion a contornos
# ===========================================================================
def introduccion_contornos():
    """Detecta contornos sobre una imagen general. Cambia el umbral para ver
    como crece o decrece la cantidad de contornos detectados.
    """
    img = imread_u(RUTA_IMAGEN)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir {RUTA_IMAGEN}")
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(15, 5))
    for i, umbral in enumerate((80, 120, 180), start=1):
        _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(
            binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        resultado = img.copy()
        cv2.drawContours(resultado, contornos, -1, (0, 255, 0), 2)
        plt.subplot(1, 3, i)
        plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        plt.title(f"Umbral={umbral}  Contornos={len(contornos)}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 2. Conteo de monedas
# ===========================================================================
def conteo_monedas():
    img = imread_u(RUTA_MONEDAS)
    if img is None:
        print(f"[2] No se encontro {RUTA_MONEDAS}, se omite.")
        return
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # OTSU + INV: obtenemos blob blanco para cada moneda con umbral automatico
    _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Limpieza morfologica para que cada moneda sea una sola componente
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Filtramos contornos demasiado pequenos (ruido)
    monedas = [c for c in contornos if cv2.contourArea(c) > 300]

    resultado = img.copy()
    cv2.drawContours(resultado, monedas, -1, (0, 255, 0), 2)
    for i, c in enumerate(monedas, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(
            resultado, str(i), (x + w // 2 - 10, y + h // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(binaria, cmap="gray")
    plt.title("Mascara binaria (OTSU + morfologia)")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    plt.title(f"Monedas detectadas: {len(monedas)}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 3. Tornillos: discriminacion por area + ensayo de tamanos de kernel
# ===========================================================================
def conteo_tornillos():
    img = imread_u(RUTA_TORNILLOS)
    if img is None:
        print(f"[3] No se encontro {RUTA_TORNILLOS}, se omite.")
        return
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria_base = cv2.threshold(
        gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    plt.figure(figsize=(16, 5))
    for i, k in enumerate(((3, 3), (7, 7), (15, 15)), start=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k)
        binaria = cv2.morphologyEx(binaria_base, cv2.MORPH_OPEN, kernel)
        contornos, _ = cv2.findContours(
            binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        resultado = img.copy()
        contador = 0
        for c in contornos:
            if cv2.contourArea(c) > 500:
                contador += 1
                cv2.drawContours(resultado, [c], -1, (255, 0, 0), 2)
        plt.subplot(1, 3, i)
        plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        plt.title(f"Kernel {k} -> {contador} tornillos")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 4 + 5. Centroide y discriminacion por area
# ===========================================================================
def centroides_y_discriminacion():
    """Sobre la imagen de tornillos: marca el centroide de cada contorno y
    los pinta de verde (grandes) o rojo (pequenos).
    """
    img = imread_u(RUTA_TORNILLOS)
    if img is None:
        print(f"[4] No se encontro {RUTA_TORNILLOS}, se omite.")
        return
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    resultado = img.copy()
    centro_imagen = (img.shape[1] // 2, img.shape[0] // 2)
    mejor_dist = float("inf")
    mejor_centro = None

    for c in contornos:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        # Color por tamano
        color = (0, 255, 0) if area > 1000 else (0, 0, 255)
        cv2.drawContours(resultado, [c], -1, color, 2)

        # Centroide via momentos
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(resultado, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(
            resultado, f"({cx},{cy})", (cx + 8, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1,
        )
        d = (cx - centro_imagen[0]) ** 2 + (cy - centro_imagen[1]) ** 2
        if d < mejor_dist:
            mejor_dist = d
            mejor_centro = (cx, cy)

    if mejor_centro is not None:
        cv2.circle(resultado, mejor_centro, 12, (0, 255, 255), 2)
        cv2.putText(
            resultado, "MAS CENTRADO", (mejor_centro[0] + 14, mejor_centro[1] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
        )

    mostrar_bgr(
        resultado,
        "Centroides (azul) | Verde=area>1000  Rojo=area<=1000  Amarillo=mas centrado",
    )


# ===========================================================================
# 6. pointPolygonTest
# ===========================================================================
def punto_dentro_contorno():
    img = imread_u(RUTA_MONEDAS)
    if img is None:
        print(f"[6] No se encontro {RUTA_MONEDAS}, se omite.")
        return
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contornos = [c for c in contornos if cv2.contourArea(c) > 300]

    # Genero algunos puntos: el centro de la imagen y 4 esquinas medias
    h, w = img.shape[:2]
    puntos = [
        (w // 2, h // 2),
        (w // 4, h // 4),
        (3 * w // 4, h // 4),
        (w // 4, 3 * h // 4),
        (3 * w // 4, 3 * h // 4),
    ]

    resultado = img.copy()
    for p in puntos:
        dentro = False
        for c in contornos:
            if cv2.pointPolygonTest(c, (float(p[0]), float(p[1])), False) >= 0:
                dentro = True
                break
        color = (0, 255, 0) if dentro else (0, 0, 255)
        cv2.circle(resultado, p, 8, color, -1)
        etiqueta = "DENTRO" if dentro else "FUERA"
        cv2.putText(
            resultado, etiqueta, (p[0] + 10, p[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        print(f"[6] Punto {p}: {etiqueta}")

    mostrar_bgr(resultado, "pointPolygonTest: verde=dentro, rojo=fuera")


# ===========================================================================
# 7. Comparacion de formas con matchShapes
# ===========================================================================
def _mayor_contorno(ruta):
    img = imread_u(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return img, None
    return img, max(contornos, key=cv2.contourArea)


def comparar_engranajes():
    img1, c1 = _mayor_contorno(RUTA_ENGRANAJE_1)
    img2, c2 = _mayor_contorno(RUTA_ENGRANAJE_2)
    if c1 is None or c2 is None:
        print("[7] Faltan engranaje1.jpg / engranaje2.jpg, se omite.")
        return

    # Auto comparacion (debe dar ~0)
    valor_self = cv2.matchShapes(c1, c1, cv2.CONTOURS_MATCH_I1, 0)
    valor = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
    print(f"[7] matchShapes engranaje1 vs engranaje1 = {valor_self:.6f} (referencia)")
    print(f"[7] matchShapes engranaje1 vs engranaje2 = {valor:.6f}")
    print("    -> cercano a 0: muy parecidos. Mas alto: mas diferentes.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap="gray")
    plt.title("engranaje1")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap="gray")
    plt.title("engranaje2")
    plt.axis("off")
    plt.suptitle(f"matchShapes = {valor:.6f}")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 8. Videovigilancia con HOG + zona segura
# ===========================================================================
def videovigilancia(max_frames=100, mostrar_cada=20):
    """Recorre el video y dibuja personas detectadas. Si alguien entra en la
    zona segura azul, dibuja ALERTA en rojo. Muestra unas pocas capturas.
    """
    if not os.path.exists(RUTA_VIDEO):
        print(f"[8] No se encontro {RUTA_VIDEO}, se omite.")
        return

    video = video_capture_u(RUTA_VIDEO)
    if video is None:
        print("[8] No se pudo abrir el video.")
        return
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Zona segura: rectangulo azul. Se ajusta al primer frame.
    ret, primer_frame = video.read()
    if not ret:
        print("[8] No se pudo leer el video.")
        video.release()
        return
    h, w = primer_frame.shape[:2]
    zona_x1, zona_y1 = w // 4, h // 4
    zona_x2, zona_y2 = 3 * w // 4, 3 * h // 4

    capturas = []
    frame_idx = 0
    # Reposicionamos para procesar desde el principio
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while frame_idx < max_frames:
        ret, frame = video.read()
        if not ret:
            break
        frame_idx += 1

        # Reducir el frame acelera HOG sin perder mucha precision
        escala = 1.0
        if max(frame.shape[:2]) > 640:
            escala = 640 / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=escala, fy=escala)
            h, w = frame.shape[:2]
            zona_x1, zona_y1 = w // 4, h // 4
            zona_x2, zona_y2 = 3 * w // 4, 3 * h // 4

        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cx, cy = x + bw // 2, y + bh // 2
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            if zona_x1 < cx < zona_x2 and zona_y1 < cy < zona_y2:
                cv2.putText(
                    frame, "ALERTA", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

        cv2.rectangle(frame, (zona_x1, zona_y1), (zona_x2, zona_y2), (255, 0, 0), 2)
        cv2.putText(
            frame, "ZONA SEGURA", (zona_x1 + 5, zona_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
        )

        if frame_idx % mostrar_cada == 0:
            capturas.append(frame.copy())

    video.release()

    if not capturas:
        capturas = [frame]

    n = len(capturas)
    plt.figure(figsize=(6 * n, 6))
    for i, f in enumerate(capturas, start=1):
        plt.subplot(1, n, i)
        plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame ~{i * mostrar_cada}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Programa principal
# ===========================================================================
def main():
    print("=" * 60)
    print("PRACTICA 05 - Contornos y aplicaciones")
    print("=" * 60)

    introduccion_contornos()
    conteo_monedas()
    conteo_tornillos()
    centroides_y_discriminacion()
    punto_dentro_contorno()
    comparar_engranajes()
    videovigilancia()

    print("\n[OK] Practica 5 finalizada.")


if __name__ == "__main__":
    main()
