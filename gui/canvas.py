import tkinter as tk


class NumberGuesserGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Number Guesser")

        self.canvas = tk.Canvas(
            root,
            width=280,
            height=280,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=20)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=(0, 20))

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear
        )
        clear_button.pack(side=tk.LEFT, padx=5)

        guess_button = tk.Button(
            button_frame,
            text="Guess",
            command=self.guess
        )
        guess_button.pack(side=tk.LEFT, padx=5)

        self.result = tk.Label(
            root,
            text="Draw a number",
            font=("Arial", 20)
        )
        self.result.pack(pady=(0, 20))

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=15,
            capstyle=tk.ROUND,
            smooth=True
        )

        self.last_x = event.x
        self.last_y = event.y

    def clear(self):
        self.canvas.delete("all")
        self.result.config(text="Draw a number")

    def guess(self):
        self.result.config(text="idk")


root = tk.Tk()

app = NumberGuesserGUI(root)

root.mainloop()
