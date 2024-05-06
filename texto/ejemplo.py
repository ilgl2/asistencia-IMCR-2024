import easyocr
reader = easyocr.Reader(['en', 'es'])

result1 = reader.readtext('imágenes/autovía.jpg', detail=0, paragraph=True)
result2 = reader.readtext('imágenes/calle.jpg', detail=0, paragraph=True)
result3 = reader.readtext('imágenes/metro.jpg', detail=0, paragraph=True)


print(result1, result2, result3)

