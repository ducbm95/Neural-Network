import Tkinter as tk
from PIL import ImageGrab
import recognization
# import image_processing as ip

class PaintBox(tk.Frame):
    
    def __init__(self):
        tk.Frame.__init__(self)
        self.pack(expand=tk.YES, fill=tk.BOTH)
        self.master.title("Handwritten Digit Detection")
        self.master.geometry("700x500")
        
        self.canvas = tk.Canvas(self)
        self.canvas.config(width=500, height=500, background="white")
        self.canvas.pack(side=tk.LEFT)
        
        self.button_ok = tk.Button(self, text="Detect", command=self.detect_digit)
        self.button_ok.config(width=200)
        self.button_ok.pack(side=tk.TOP)
        
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.config(width=200)
        self.button_clear.pack(side=tk.TOP)
        
        self.message = tk.Label(self, text="Drag the mouse to draw")
        self.message.pack(side=tk.BOTTOM)
        
        self.canvas.bind("<Button-1>", self.mouse_clicked)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.x = 0
        self.y = 0
        
    def mouse_clicked(self, event):
        """
        this event is called when left mouse is clicked.
        """
        self.x = event.x
        self.y = event.y
        
    def draw(self, event):
        """
        This event is called when mouse is hold and moved.
        """
        self.canvas.create_line(self.x, self.y, event.x, event.y, width=6)
        self.x = event.x
        self.y = event.y
        
    def clear_canvas(self):
        self.canvas.delete("all")
        
    def detect_digit(self):
        # get screen of draw
        self.crop_image(self.canvas)
        
#         img = ip.resize_image("temp.png")
#         img = ip.normalize_image(img)
#         out = ip.detect_image(img)
#         self.message.config(text=out)
        
        recognization.reg()

    def crop_image(self, widget):
        """
        Crop an image of widget and save it to temp.png
        """
        x = self.winfo_rootx() + widget.winfo_x()
        y = self.winfo_rooty() + widget.winfo_y()
        x1 = x + widget.winfo_width()
        y1 = y + widget.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save("temp.png")
        
def main():
    PaintBox().mainloop()
    
main()
