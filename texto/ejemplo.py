import easyocr
reader = easyocr.Reader(['en', 'es'])

result1 = reader.readtext('./texto/data/autovia.jpg', detail=0, paragraph=True)
result2 = reader.readtext('./texto/data/calle.jpg', detail=0, paragraph=True)
result3 = reader.readtext('./texto/data/metro.jpg', detail=0, paragraph=True)

print(result1, result2, result3)

