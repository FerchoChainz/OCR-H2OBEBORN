import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from pdf2image import convert_from_path
import os
import pandas as pd
from easyocr import Reader
import re
import shutil

# Constantes de configuración
MAX_DIGITS = 14
MIN_DIGITS = 12
MORPH_KERNEL_SIZE = (50, 1)
FONT_SIZE = 10
OCR_LANGUAGES = ['en', 'es']

# Rutas de archivos y directorios
CONFIG = {
    'pdf_path': '/Users/lazaroestrada/Desktop/Develop/H2OBEBORN/REQ-TEST.pdf',
    'output_dir': '/Users/lazaroestrada/Desktop/Develop/H2OBEBORN/images',
    'processed_dir': '/Users/lazaroestrada/Desktop/Develop/H2OBEBORN/images-process',
    'font_path': '/Users/lazaroestrada/Desktop/Develop/H2OBEBORN/calibri.ttf',
    'csv_path': '/Users/lazaroestrada/Desktop/Develop/H2OBEBORN/detected_text.csv',
    'csv_output_path': '/Users/lazaroestrada/Desktop/Develop/H2OBEBORN/processed_text.csv',
}

# Crear las carpetas 'images' y 'images-process' si no existen
def create_dirs():
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
    if not os.path.exists(CONFIG['processed_dir']):
        os.makedirs(CONFIG['processed_dir'])

# Inicializar el lector de EasyOCR
reader = Reader(OCR_LANGUAGES)

# Lista para almacenar los datos extraídos
data = []

# Funciones auxiliares
def write_text(text, x, y, img, font_path, color=(50, 50, 255), font_size=FONT_SIZE, position="top"):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # Ajuste de posición del texto
    if position == "top":
        draw.text((x, y - font_size), text, font=font, fill=color)
    elif position == "bottom":
        draw.text((x, y + font_size), text, font=font, fill=color)
    
    return np.array(img_pil)

def box_coordinates(box):
    (lt, rt, br, bl) = box
    lt = (int(lt[0]), int(lt[1]))
    rt = (int(rt[0]), int(rt[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    return lt, rt, br, bl

def draw_img(img, lt, br, color=(200, 255, 0), thickness=2):
    cv2.rectangle(img, lt, br, color, thickness)
    return img

def replace_to_number(text):
    # Reemplazar letras 'O' y 'U' con '0' para asegurar que son valores numéricos
    text = re.sub(r'O(?=\d)|(?<=\d)O|(?<=\d)O(?=\d)', '0', text)
    text = text.replace('OOO', '000').replace('OO', '00').replace('00O', '000').replace('U00', '000')
    return text

def safe_convert_to_string(value):
    return str(value) if not pd.isnull(value) else ''

def detect_table_and_apply_ocr(image_path, processed_dir, page_num):
    print('APLICANDO OCR')
    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    
    if large_contours:
        contours = sorted(large_contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        table_img = img[y:y+h, x:x+w]
        results = reader.readtext(table_img)

        for (box, text, probability) in results:
            lt, rt, br, bl = box_coordinates(box)
            text_upper = text.upper()
            text_corrected = replace_to_number(text_upper)
            
            # Dibuja el rectángulo y escribe el texto
            table_img = draw_img(table_img, lt, br)
            table_img = write_text(text_corrected, lt[0], lt[1], table_img, CONFIG['font_path'], position="top")
            
            # Calcula el punto medio
            mid_x = (lt[0] + br[0]) // 2
            mid_y = (lt[1] + br[1]) // 2
            midpoint_text = f"(x: {mid_x}, y: {mid_y})"
            
            # Escribe el punto medio en la parte inferior del bounding box
            table_img = write_text(midpoint_text, mid_x - 30, br[1], table_img, CONFIG['font_path'], position="bottom")
            
            # Agregar al DataFrame incluyendo la posición
            data.append([f'pagina{page_num}.png', text_corrected, probability, midpoint_text])

        processed_image_path = os.path.join(processed_dir, f'tabla_pagina_procesada_{page_num}.png')
        cv2.imwrite(processed_image_path, table_img)
    
    os.remove(image_path)
    print(f"Imagen {image_path} eliminada")

    # Crear el DataFrame con la nueva columna POSITION
    df = pd.DataFrame(data, columns=['Imagen', 'Texto detectado', 'Precisión', 'POSITION'])
    df.to_csv(CONFIG['csv_path'], index=False)


def group_digits(csv_path, csv_output_path):
    # Leer el archivo CSV generado por OCR
    df = pd.read_csv(csv_path)
    df['CLAVE'] = None  # Añadir la columna CLAVE para almacenar las secuencias agrupadas
    df['CANT'] = None  # Añadir la columna CANT para almacenar el valor adicional
    current_sequence = ''  # Para almacenar la secuencia de dígitos
    start_index = None  # Para recordar el índice inicial de la secuencia
    base_y = None  # Para almacenar la coordenada Y inicial
    final_x = None  # Para almacenar la coordenada X del último dígito de la secuencia

    for index, row in df.iterrows():
        text = safe_convert_to_string(row['Texto detectado'])
        position = row['POSITION']
        
        # Extraer coordenadas X y Y de la posición
        match = re.search(r'x: (\d+), y: (\d+)', position)
        if match:
            x_coord, y_coord = int(match.group(1)), int(match.group(2))

            if re.match(r'^\d+$', text):  # Solo procesamos si el texto es puramente numérico
                # Si es el inicio de una secuencia con 010 o 040 y la secuencia actual está vacía
                if re.match(r'^010|^040|^030', text) and len(current_sequence) == 0:
                    current_sequence = text
                    start_index = index
                    base_y = y_coord  # Guardar coordenada Y inicial para comparar
                    final_x = x_coord  # Guardar la coordenada X inicial

                # Si la coordenada Y está cerca del valor base_y y la secuencia no excede 14 dígitos
                elif len(current_sequence) > 0 and abs(y_coord - base_y) <= 20 and len(current_sequence) + len(text) <= MAX_DIGITS:
                    current_sequence += text
                    final_x = x_coord  # Actualizar la última coordenada X del dígito en la secuencia

                # Verificar si la secuencia tiene entre 12 y 14 dígitos
                if len(current_sequence) == MAX_DIGITS or (MIN_DIGITS <= len(current_sequence) < MAX_DIGITS and index == len(df) - 1):
                    # Rellenar las filas correspondientes con la secuencia agrupada
                    for i in range(start_index, index + 1):
                        df.at[i, 'CLAVE'] = current_sequence

                        found = False  # Indicador para saber si encontramos un valor a 1400 píxeles

                        # Primera búsqueda: a una distancia de 1400 píxeles
                        for j in range(index + 1, len(df)):
                            extra_text = safe_convert_to_string(df.at[j, 'Texto detectado'])
                            extra_position = df.at[j, 'POSITION']
                            match_extra = re.search(r'x: (\d+), y: (\d+)', extra_position)
                            
                            if match_extra:
                                extra_x, extra_y = int(match_extra.group(1)), int(match_extra.group(2))
                                
                                # Condición para buscar a 1400 píxeles de distancia
                                if abs(extra_y - base_y) <= 20 and (extra_x - final_x) >= 1400 and re.match(r'^\d+$', extra_text):
                                    df.at[start_index, 'CANT'] = extra_text
                                    found = True  # Actualizamos el indicador si encontramos un valor a 1400 píxeles
                                    break  # Detener la búsqueda una vez encontrado el valor a 1400 píxeles

                        # Segunda búsqueda: solo si no se encontró un valor a 1400 píxeles, buscar a 1300 píxeles
                        if not found:
                            for j in range(index + 1, len(df)):
                                extra_text = safe_convert_to_string(df.at[j, 'Texto detectado'])
                                extra_position = df.at[j, 'POSITION']
                                match_extra = re.search(r'x: (\d+), y: (\d+)', extra_position)
                                
                                if match_extra:
                                    extra_x, extra_y = int(match_extra.group(1)), int(match_extra.group(2))
                                    
                                    # Condición para buscar a 1300 píxeles de distancia
                                    if abs(extra_y - base_y) <= 20 and (extra_x - final_x) >= 1300 and re.match(r'^\d+$', extra_text):
                                        df.at[start_index, 'CANT'] = extra_text
                                        break  # Detener la búsqueda una vez encontrado el valor a 1300 píxeles

                    # Reiniciar la secuencia y las coordenadas base
                    current_sequence = ''
                    start_index = None
                    base_y = None
                    final_x = None
            else:
                # Si no es un número, reiniciamos la secuencia
                current_sequence = ''
                start_index = None
                base_y = None
                final_x = None

    # Filtrar filas donde CLAVE no es nulo y contiene solo valores numéricos
    df = df.dropna(subset=['CLAVE'])
    df['CLAVE'] = df['CLAVE'].apply(lambda x: x if re.match(r'^\d+$', str(x)) else None)
    df = df.dropna(subset=['CLAVE']).drop_duplicates(subset=['CLAVE'])

    # Guardar solo las columnas CLAVE y CANT en el CSV final
    df[['CLAVE', 'CANT']].to_csv(csv_output_path, index=False)
    print(f"Archivo CSV generado con campos CLAVE y CANT en: {csv_output_path}")

def clean_up():
    paths_to_clean = [CONFIG['processed_dir'], CONFIG['csv_path'], CONFIG['output_dir']]
    for path in paths_to_clean:
        if os.path.exists(path):
            # Usa shutil.rmtree si es un directorio, y os.remove si es un archivo
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    print("Archivos y carpetas temporales eliminados")


# Convertir el PDF en imágenes
def convert_pdf_to_images(pdf_path, output_dir):
    pages = convert_from_path(pdf_path, 300)
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f'pagina{i+1}.png')
        page.save(image_path, 'PNG')
        detect_table_and_apply_ocr(image_path, CONFIG['processed_dir'], i+1)


# Ejecución principal
def main():
    create_dirs()
    convert_pdf_to_images(CONFIG['pdf_path'], CONFIG['output_dir'])
    group_digits(CONFIG['csv_path'], CONFIG['csv_output_path'])
    clean_up()

if __name__ == '__main__':
    main()