import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
from neural_network import network
from neural_network.network import NeuralNetwork


class NumberGuesserGUI:
    CANVAS_SIZE = 280
    OUTPUT_SIZE = 28
    STROKE_WIDTH = 15

    def __init__(self, root):
        self.network = NeuralNetwork()
        self.network.load("models/digit_model")

        self.root = root
        self.root.title("Number Guesser")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(padx=20, pady=20)

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

        self.canvas.create_oval(
            event.x - self.STROKE_WIDTH // 2,
            event.y - self.STROKE_WIDTH // 2,
            event.x + self.STROKE_WIDTH // 2,
            event.y + self.STROKE_WIDTH // 2,
            fill="white",
            outline="white",
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

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")
        self.result.config(text="Draw a number")

    def guess(self):
        image = self.get_image()

        pixels = np.array(image, dtype=np.float32)

        pixels /= 255.0

        network_input = pixels.reshape(1, 28, 28)

        prediction = self.network.predict(network_input)[0]

        print("Prediction: ", prediction)

        self.result.config(text=f"You drew the {prediction} horribly ew fuck you")

    def get_image(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()

        x2 = x + self.canvas.winfo_width()
        y2 = y + self.canvas.winfo_height()

        image = ImageGrab.grab((x, y, x2, y2))

        image = image.convert("L")

        threshold = 20

        bbox = image.point(lambda pixel: 255 if pixel > threshold else 0).getbbox()

        if bbox is None:
            return Image.new("L", (28, 28), 0)

        image = image.crop(bbox)

        width, height = image.size

        size = max(width, height)

        padding = int(size * 0.20)

        size += padding * 2

        square = Image.new("L", (size, size), 0)

        x_offset = (size - width) // 2
        y_offset = (size - height) // 2

        square.paste(image, (x_offset, y_offset))

        digit = square.resize((20, 20), Image.Resampling.LANCZOS)

        final_image = Image.new("L", (28, 28), 0)

        final_image.paste(digit, (4, 4))

        final_image.save("data/digit.png")

        return final_image
