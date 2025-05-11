import cv2
import numpy as np

def decode_image(image_bytes: bytes, fmt: str) -> np.ndarray:
    img_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image

def preprocess_face(img: np.ndarray, model_name: str) -> np.ndarray:
    """
    Preprocesses the face image for ONNX models based on specific input size and channel expectations.
    """
    print(f"Shape of input img: {img.shape}")

    # Define model-specific input sizes
    model_input_sizes = {
        "arcface": (112, 112),
        "glintr100": (112, 112),
        "buffalo": (192, 192),
        "1k3d68": (192, 192),
        "2d106det": (192, 192),
        "scrfd_10g_bnkps": (640, 640),
        "genderage": (96, 96), #genderage ONNX model expects input size (96, 96)
    }

    model_key = model_name.lower()
    if model_key not in model_input_sizes:
        raise ValueError(f"Unknown model name '{model_name}'. Please check spelling or add to model_input_sizes.")

    resize_size = model_input_sizes[model_key]
    resized = cv2.resize(img, resize_size)

    # Handle grayscale or alpha channel cases
    if len(resized.shape) == 2 or resized.shape[2] == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)

    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_img / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(transposed.astype(np.float32), axis=0)

    return input_tensor
