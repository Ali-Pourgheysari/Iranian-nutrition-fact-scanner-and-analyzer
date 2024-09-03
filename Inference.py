import tkinter as tk
from tkinter import filedialog, messagebox
import easyocr
from easyocr.detection import get_detector, get_textbox
import torch
import cv2
import os
import random
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from Analyse import isMatched, Analyse, Find_certificate_number
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Nutrition fact label scanner & analyser")
        self.root.geometry("600x400")
        self.root.configure(bg="#2c3e50")

        # Set up custom fonts
        self.custom_font = ("Helvetica", 14, "bold")
        self.button_font = ("Helvetica", 12, "bold")
        self.label_font = ("Helvetica", 10)

        # Create a frame for the main content
        self.main_frame = tk.Frame(root, bg="#34495e", bd=5)
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=500, height=300)

        # Create a label
        self.title_label = tk.Label(self.main_frame, text="Nutrition fact label scanner & analyser", font=self.custom_font, fg="white", bg="#34495e")
        self.title_label.pack(pady=20)

        # Select Image Directory button
        self.select_img_button = tk.Button(self.main_frame, text="Select Image Directory",
                                           font=self.button_font, bg="#3498db", fg="white", width=25, height=2, cursor="hand2",
                                           command=self.select_img_directory)
        self.select_img_button.pack(pady=10)

        # Select Output Directory button
        self.select_output_button = tk.Button(self.main_frame, text="Select Output Directory",
                                              font=self.button_font, bg="#e74c3c", fg="white", width=25, height=2, cursor="hand2",
                                              command=self.select_output_directory)
        self.select_output_button.pack(pady=10)

        # Start Processing button
        self.start_button = tk.Button(self.main_frame, text="Start Processing",
                                      font=self.button_font, bg="#2ecc71", fg="white", width=25, height=2,
                                      command=self.start_processing, state=tk.DISABLED)
        self.start_button.pack(pady=20)


        # Set paths to None
        self.img_path = None
        self.output_path = None

        # Create a label to hold the spinning animation
        self.spinner_label = tk.Label(self.main_frame, bg="#34495e")
        self.spinner_label.pack(pady=20)
        
        # Load spinning GIF frames
        self.spinner_frames = [ImageTk.PhotoImage(Image.open(f"./Data/Spinner/{i}.png")) for i in range(1, 22)]
        self.current_frame = 0
        self.animating = False

    def select_img_directory(self):
        self.img_path = filedialog.askdirectory(title="Select the directory containing images")
        if self.img_path:
            self.check_ready_to_start()

    def select_output_directory(self):
        self.output_path = filedialog.askdirectory(title="Select the output directory") + '/'
        if self.output_path:
            self.check_ready_to_start()

    def check_ready_to_start(self):
        if self.img_path and self.output_path:
            self.start_button.config(state=tk.NORMAL, cursor="hand2")

    def start_animation(self):
        if self.animating:
            # Update the spinner frame
            self.spinner_label.config(image=self.spinner_frames[self.current_frame])
            self.current_frame = (self.current_frame + 1) % len(self.spinner_frames)

            # Schedule the next frame update
            self.root.after(100, self.start_animation)
    
    def stop_animation(self):
        self.animating = False
        self.spinner_label.config(image="")

    def start_processing(self):
        # Hide all buttons
        self.select_img_button.pack_forget()
        self.select_output_button.pack_forget()
        self.start_button.pack_forget()

        # Start the spinner animation
        self.animating = True
        self.start_animation()

        # Start processing in a separate thread to keep the GUI responsive
        processing_thread = threading.Thread(target=self.run_processing)
        processing_thread.start()

    def resize(self, image, min_height=1280):
        # Resaize and keep aspect ratio
        h, w = image.shape[:2]
        scale = min_height / h
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image
    
    def run_processing(self):
        try:
            # Load the custom YOLO model for object detection
            label_detection_model = YOLO("Models/Label_detection.pt") 

            # Load the custom CRAFT model
            craft_save_pth = torch.load('Models/CRAFT_clr_amp_34000_augmented.pth', map_location='cpu')
            craft_model = craft_save_pth["craft"]

            # Save the models separately for the detector and recognizer
            torch.save(craft_model, "Models/CRAFT_detector.pth")

            # Initialize EasyOCR reader without loading default models
            reader = easyocr.Reader(
                lang_list=["fa"],
                detector=False,
                user_network_directory='./Models',
                model_storage_directory='./Models',
                recog_network='Ocr_best'
            )

            # Correctly assign custom models to reader
            reader.get_detector = get_detector
            reader.get_textbox = get_textbox
            reader.detector = reader.initDetector("Models/CRAFT_detector.pth")

            # Get list of images
            img_lst = os.listdir(self.img_path)
            random.shuffle(img_lst)

            # Process each image
            for i in img_lst:
                img_file = os.path.join(self.img_path, i)
                label_detection_model = YOLO("Models/Label_detection.pt")

                # Perform object detection
                yolo_results = label_detection_model(img_file)
                
                for yolo_result in yolo_results:
                    # Get the bounding box and label of the detected object
                    bbox = yolo_result.boxes.xyxy[0].cpu().numpy()

                    # Read and crop the image based on the detected object
                    image = cv2.imread(img_file)
                    image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    image = self.resize(image)
                    result = reader.readtext(image)
                    
                    energy_q1 = -1
                    energy_y1 = -1
                    
                    for (bbox, text, prob) in result:
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        image = cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (0, 255, 0), 5)

                        if text[0] < 'z' and text[0] > 'A' or text[0] >= '0' and text[0] <= '9':
                            cv2.putText(image, text, (int(top_left[0]), int(top_left[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        else:
                            # put persian text on image
                            reshaped_text = arabic_reshaper.reshape(text) # correct its shape
                            bidi_text = get_display(reshaped_text) # correct its direction
                            img_pil = Image.fromarray(image)
                            draw = ImageDraw.Draw(img_pil)
                            font = ImageFont.truetype('./Data/Fonts/BYekan.ttf', 30)
                            draw.text((int(top_left[0]), int(top_left[1]) - 15), bidi_text, font=font, fill=(255, 0, 0))
                            image = np.array(img_pil)

                            if isMatched(text, 'انرژی'):
                                energy_q1 = (bottom_right[1] - top_left[1]) / 4
                                energy_y1 = top_left[1]
                            
                        file_name = i.replace('.jpg', '')
                        Find_certificate_number(bbox, text, image.shape[1], image.shape[0], self.output_path, file_name)
                            
                    if energy_q1 != -1:
                        file_name = i.replace('.jpg', '')
                        Analyse(result, energy_q1, energy_y1, self.output_path, file_name)
                    
                    # Write the annotated image to the output directory
                    output_file = os.path.join(self.output_path, i)
                    cv2.imwrite(output_file, image)

            # Show a message or update the UI
            self.root.after(0, lambda: messagebox.showinfo("Processing Complete", "Image processing is complete!"))
        finally:
            self.root.after(0, self.stop_animation)
        
        # Restart
        self.__init__(self.root)



# Initialize the main window
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
