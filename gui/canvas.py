import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from neural_network.network import NeuralNetwork

from neural_network.preprocessing import preprocess_image


class NumberGuesserGUI:
    CANVAS_SIZE = 280
    STROKE_WIDTH = 10

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

        network_input = preprocess_image(self.image)

        network_input = network_input.reshape(1, 64 * 64)

        probabilities = self.network.forward(network_input)

        prediction = np.argmax(probabilities[0])

        print("Probabilities:")

        for digit, probability in enumerate(probabilities[0]):
            print(f"{digit}: {probability:.4f}")

        print("Prediction:", prediction)

        confidence = probabilities[0][prediction] * 100

        self.result.config(text=(f"You drew {prediction} ({confidence:.1f}%)"))


if __name__ == "__main__":
    root = tk.Tk()

    app = NumberGuesserGUI(root)

    root.mainloop()
