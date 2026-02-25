import io
from PIL import Image

def validate_image_file(image_bytes: bytes) -> bool:
    try:
        file = io.BytesIO(image_bytes)
        img = Image.open(file)
        img.verify()  
        file.close()
        img.close()
        return True
    except Exception:
        return False