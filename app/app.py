import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import ImageTk,Image
import cv2 as cv
from processor import Processor

class ImageUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Uploader")
        self.root.geometry("800x600")

        # Label to display image
        self.input_image_label = Label(self.root, text="No Image Selected", width=25, height=15, bg="gray")
        self.input_image_label.pack(pady=20)

        # Button to upload image
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        # Output
        self.output_image_label = Label(self.root, text="No Output", width=40, height=10, bg="gray")
        self.output_image_label.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            img = cv.imread(file_path)

            self.show_input_img(img)

            #### Add Logic Here

            Processor.process(img)
            ## and complete adding logic



            #### End of Logic
            #### Output

            ## Write Logic that simulates your output here 

    def show_input_img(self,img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img_rgb)  # Convert to PIL Image

        img_pil = img_pil.resize((300, 200), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)  # Convert to Tkinter format

        # Update the label with the image
        self.input_image_label.config(image=img_tk,height=300,width=200, text="")
        self.input_image_label.image = img_tk

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
