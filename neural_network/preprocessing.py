import numpy as np
from PIL import Image

OUTPUT_SIZE = 64


def preprocess_image(path_or_image):
    if isinstance(path_or_image, str):
        image = Image.open(path_or_image).convert("L")
    else:
        image = path_or_image.convert("L")

    pixels = np.array(image)

    corner_size = max(1, min(image.size) // 10)

    corners = np.concatenate(
        [
            pixels[:corner_size, :corner_size].flatten(),
            pixels[:corner_size, -corner_size:].flatten(),
            pixels[-corner_size:, :corner_size].flatten(),
            pixels[-corner_size:, -corner_size:].flatten(),
        ]
    )

    if np.mean(corners) > 127:
        pixels = 255 - pixels

    image = Image.fromarray(pixels.astype(np.uint8), mode="L")

    bbox = image.getbbox()

    if bbox is None:
        return np.zeros(OUTPUT_SIZE * OUTPUT_SIZE, dtype=np.float32)

    image = image.crop(bbox)

    width, height = image.size

    size = max(width, height)

    square = Image.new("L", (size, size), 0)

    x_offset = (size - width) // 2
    y_offset = (size - height) // 2

    square.paste(image, (x_offset, y_offset))

    padded_size = max(int(size * 1.4), size + 1)

    padded = Image.new("L", (padded_size, padded_size), 0)

    offset = (padded_size - size) // 2

    padded.paste(square, (offset, offset))

    final_image = padded.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)

    result = np.array(final_image, dtype=np.float32) / 255.0

    return result.reshape(OUTPUT_SIZE * OUTPUT_SIZE)
