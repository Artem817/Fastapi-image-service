import hashlib

def get_image_hash(image_bytes: bytes) -> str:
    """Calculate a hash for the given image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()