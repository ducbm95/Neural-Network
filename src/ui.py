import Tkinter as tk
from PIL import ImageGrab
import cv2
from sklearn.externals import joblib
from classifier import Classifier

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
        
        self.clf = joblib.load("classifier.pkl")
        
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
        self.regcognize()

    def crop_image(self, widget):
        """
        Crop an image of widget and save it to temp.png
        """
        x = self.winfo_rootx() + widget.winfo_x()
        y = self.winfo_rooty() + widget.winfo_y()
        x1 = x + widget.winfo_width()
        y1 = y + widget.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save("temp.png")
    
    def regcognize(self):
        im = cv2.imread("temp.png")
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        
        _, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the image
        ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get rectangles contains each contour
        rects = [cv2.boundingRect(x) for x in ctrs]
        
        for rect in rects:
            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)
            
            # Make the rectangular region around the digit
            border = int(0.3 * max(rect[2], rect[3]))
            roi = im_th[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            roi = cv2.copyMakeBorder(roi, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
            
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            
            output = self.clf.predict(roi)
            cv2.putText(im, str(int(output)), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)
        
        cv2.imshow("Result", im)
        cv2.waitKey()
        
        
def generate_classifier(self):
    x = Classifier()
    joblib.dump(x, "classifier.pkl", compress=3)
    
def main():
    PaintBox().mainloop()
    
main()
