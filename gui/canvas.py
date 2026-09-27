import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from neural_network.network import NeuralNetwork


class NumberGuesserGUI:
    CANVAS_SIZE = 280
    OUTPUT_SIZE = 28
    STROKE_WIDTH = 15

    def __init__(self, root):
        self.root = root
        self.root.title("Number Guesser")
        self.root.resizable(False, False)

        self.network = NeuralNetwork()
        self.network.load("models/digit_model.npz")

        self.canvas = tk.Canvas(
            root,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="black",
            highlightthickness=0,
        )

        self.canvas.pack(padx=20, pady=20)

        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)

        self.image_draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.start_drawing)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=(0, 20))

        clear_button = tk.Button(
            button_frame, text="Clear", width=10, command=self.clear
        )

        clear_button.pack(side=tk.LEFT, padx=5)

        guess_button = tk.Button(
            button_frame, text="Guess", width=10, command=self.guess
        )

        guess_button.pack(side=tk.LEFT, padx=5)

        self.result = tk.Label(root, text="Draw a number", font=("Arial", 20))

        self.result.pack(pady=(0, 20))

        self.last_x = None
        self.last_y = None

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

        radius = self.STROKE_WIDTH // 2

        self.canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            fill="white",
            outline="white",
        )

        self.image_draw.ellipse(
            (event.x - radius, event.y - radius, event.x + radius, event.y + radius),
            fill=255,
        )

    def draw(self, event):
        if self.last_x is None:
            return

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=self.STROKE_WIDTH,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            smooth=True,
        )

        self.image_draw.line(
            (self.last_x, self.last_y, event.x, event.y),
            fill=255,
            width=self.STROKE_WIDTH,
        )

        radius = self.STROKE_WIDTH // 2

        self.image_draw.ellipse(
            (event.x - radius, event.y - radius, event.x + radius, event.y + radius),
            fill=255,
        )

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")

        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)

        self.image_draw = ImageDraw.Draw(self.image)

        self.result.config(text="Draw a number")

    def guess(self):
        image = self.get_image()

        pixels = np.array(image, dtype=np.float32)

        pixels /= 255.0

        print("\nGUI IMAGE:")

        for row in pixels:
            print("".join("██" if pixel > 0.01 else "  " for pixel in row))

        print("\nImage sum:", pixels.sum())

        print("Image mean:", pixels.mean())

        print("Non-zero pixels:", np.count_nonzero(pixels))

        network_input = pixels.reshape(1, 784)

        probabilities = self.network.forward(network_input)

        prediction = np.argmax(probabilities[0])

        print("\nProbabilities:")

        for digit, probability in enumerate(probabilities[0]):
            print(f"{digit}: {probability:.4f}")

        print("Prediction:", prediction)

        self.result.config(text=f"You drew the {prediction}")

    def get_image(self):
        image = self.image.copy()

        bbox = image.getbbox()

        if bbox is None:
            return Image.new("L", (28, 28), 0)

        image = image.crop(bbox)

        print("Cropped size:", image.size)

        width, height = image.size

        size = max(width, height)

        square = Image.new("L", (size, size), 0)

        x_offset = (size - width) // 2
        y_offset = (size - height) // 2

        square.paste(image, (x_offset, y_offset))

        padded_size = int(size * 1.4)

        padded = Image.new("L", (padded_size, padded_size), 0)

        offset = (padded_size - size) // 2

        padded.paste(square, (offset, offset))

        final_image = padded.resize((28, 28), Image.Resampling.LANCZOS)

        final_image.save("data/digit.png")

        return final_image
