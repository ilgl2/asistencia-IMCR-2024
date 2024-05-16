import easyocr
reader = easyocr.Reader(['en', 'es'])

def read(file):
    transcription = reader.readtext(file.getvalue(), detail=0, paragraph=True)
    print(transcription)
    transcription = " ".join(transcription)
    return transcription