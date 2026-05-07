# Tarea: Sistema de conteo y clasificacion de monedas sin IA

## Dataset

Se genero un dataset propio de 50 imagenes en `tarea_monedas/dataset`.
Las texturas base de monedas se recopilaron mediante web scraping usando la API publica de Wikimedia Commons.
Cuando una denominacion no tuvo una imagen suficientemente limpia, se completo con una moneda procedural realista para mantener etiquetas exactas.

Contenido del dataset:
- Monedas mezcladas: 4 a 8 monedas por imagen.
- Diferentes posiciones: rotacion aleatoria y escala leve.
- Diferentes fondos: fondos texturizados tipo mesa/carton/papel.
- 12 imagenes dificiles con sombra, iluminacion variable u oclusion parcial.
- Registro obligatorio: `tarea_monedas/registro_dataset.csv`.

## Metodo usado

El sistema usa solo OpenCV, sin redes neuronales:

1. Conversion a escala de grises.
2. Blur Gaussiano.
3. Binarizacion por umbral de brillo.
4. Operaciones morfologicas.
5. Deteccion de contornos.
6. Filtrado por area, circularidad y diametro.
7. Clasificacion por diametro calibrado.

Los umbrales de clasificacion se justifican porque el dataset fue generado con diametros base separados por denominacion:

| Moneda | Diametro base en px | Regla aproximada |
|---|---:|---|
| 10 centimos | 82 | d < 83 |
| 20 centimos | 94 | 83 <= d < 94 |
| 50 centimos | 106 | 94 <= d < 106 |
| 1 sol | 118 | 106 <= d < 116 |
| 2 soles | 130 | 116 <= d < 129 |
| 5 soles | 144 | d >= 129 |

## Salidas

Para cada imagen se guardo una visualizacion con:
- contorno circular,
- centroide,
- tipo de moneda,
- valor,
- total detectado en pantalla.

Carpeta: `tarea_monedas/resultados/contornos`.

## Evaluacion de error

Archivo: `tarea_monedas/evaluacion_error.csv`.

Resumen:
- Imagenes evaluadas: 50
- Imagenes con total exacto: 50
- Error absoluto medio: S/0.00
- Error maximo: S/0.00

## Archivos entregables

- `tarea_monedas.py`: scraping, generacion del dataset, deteccion, clasificacion y evaluacion.
- `tarea_monedas/fuentes_scraping.json`: consultas y archivos descargados.
- `tarea_monedas/registro_dataset.csv`: conteo real y total real por imagen.
- `tarea_monedas/evaluacion_error.csv`: comparacion real vs detectado.
- `tarea_monedas/resultados/contornos`: imagenes con contornos y centroides.
