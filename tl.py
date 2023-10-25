#This code is in main branch 
import os
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk,filedialog
import supervision as sv
import torch
import cv2
import numpy as np
from autodistill_grounded_sam import GroundedSAM
selected_option = None
global folder_path
image_paths = []
label_paths = []

def extract_bounding_boxes(image_folder_path, label_folder_path):
    global label_paths
    global image_paths
    image_listbox.delete(0, "end") 
    image_paths = []
    for filename in os.listdir(image_folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(image_folder_path, filename))
                image_listbox.insert("end", filename)
                base_name = filename.rsplit(".", 1)[0]
                labelfilename = base_name + ".txt"
                label_paths.append(os.path.join(label_folder_path, labelfilename))
        if image_listbox.size() == 0:
            dynamic_text.set("there are no images")
    show_first_image()
    
def show_rectangle(selected_image_path, label_path):
    image_canvas.delete("all")  # Clear any previously drawn rectangles
    image = Image.open(selected_image_path)
    image = image.resize((1200,700))
    image_size = image.size
    photo = ImageTk.PhotoImage(image)
    image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    image_canvas.bg_image = photo
    # image_label.config(image=photo)
    # image_label.image = photo

    with open(label_path, 'r') as f:
        for line in f:
            # Split the line to extract bounding box information for each class
            bbox_info = line.split()
            for i in range(0, len(bbox_info), 5):
                class_id = int(bbox_info[i])
                x_center = float(bbox_info[i + 1])
                y_center = float(bbox_info[i + 2])
                width = float(bbox_info[i + 3])
                height = float(bbox_info[i + 4])

                # Calculate coordinates based on image size
                x_min = int((x_center - width / 2) * image_size[0])
                y_min = int((y_center - height / 2) * image_size[1])
                x_max = int((x_center + width / 2) * image_size[0])
                y_max = int((y_center + height / 2) * image_size[1])

                # Draw the bounding box on the image label
                image_canvas.create_rectangle(x_min, y_min, x_max, y_max, outline='green', width=2)

def generate_l_paths(a):
    image_path = a + "/train/images"
    label_path = a + "/train/labels"
    extract_bounding_boxes(image_path, label_path)
    
def clear_image_and_listbox():
    blank_image = Image.new("RGB", (1200, 700), "white")
    blank_photo = ImageTk.PhotoImage(blank_image)
    image_canvas.create_image(0, 0, anchor=tk.NW, image=blank_photo)
    image_canvas.bg_image = blank_photo
    # image_label.config(image=blank_photo)
    # image_label.image = blank_photo
    image_listbox.delete(0, "end")

def run_second_code():
    global selected_option
    selected_option = option_var.get()
    if image_listbox.size() != 0:
        if selected_option:
            dynamic_text.set("Labelling.............")
            IMAGE_DIR_PATH = folder_path
            DATASET_DIR_PATH = filedialog.askdirectory()
            image_paths = sv.list_files_with_extensions(
                directory=IMAGE_DIR_PATH,
                extensions=["jpg"])
            # print('image count:', len(image_paths))
            if DATASET_DIR_PATH != '':
                from autodistill.detection import CaptionOntology
                if selected_option == 'head' :
                    ontology = CaptionOntology({
                        "Human head": "head"
                    })
                if selected_option == 'gun':
                    ontology = CaptionOntology({
                        "Gun" : "Gun"
                    })
                if selected_option == 'person':
                    ontology = CaptionOntology({
                        "Person" : "person"
                    })
                DEVICE = torch.device('cpu')
                base_model = GroundedSAM(ontology=ontology)
                dataset = base_model.label(
                    input_folder=IMAGE_DIR_PATH,
                    extension=".jpg",
                    output_folder=DATASET_DIR_PATH)

                # Define a message based on the result of your code execution
                dynamic_text.set("Images are Labelled successfully")
                tk.messagebox.showinfo("Alert","Images are Labelled successfully !....")

                clear_image_and_listbox()
                generate_l_paths(DATASET_DIR_PATH)
                return 
            else : 
                dynamic_text.set("please select an folder first!..")
                tk.messagebox.showinfo("Alert","please select an folder first!..")
                return
        else:
            dynamic_text.set("Select an Option first ")
            tk.messagebox.showinfo("Alert","Select an Option first ")
            return
    else:
        dynamic_text.set("The folder has no images! ")
        tk.messagebox.showinfo("Alert","The folder has no images !.....")
        return

def update_image(event):
    global label_paths
    global image_paths
    if len(label_paths) == 0:
        if image_listbox.curselection():  # Check if any item has been selected
            selected_index = image_listbox.curselection()[0]
            selected_image_path = image_paths[selected_index]
            image = Image.open(selected_image_path)
            image = image.resize((1200,700))
            photo = ImageTk.PhotoImage(image)
            image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            image_canvas.bg_image = photo
            # image_label.config(image=photo)
            # image_label.image = photo
        else:
            tk.messagebox.showinfo("Alert","Some unknown error!........")
    else:
        if image_listbox.curselection():
            selected_index = image_listbox.curselection()[0]
            selected_image_path = image_paths[selected_index]
            selected_label_path = label_paths[selected_index]
            print("\n ----------- \n", selected_image_path , "\n ------------ \n", selected_label_path)
            show_rectangle(selected_image_path, selected_label_path)
        else:
            tk.messagebox.showinfo("Alert","Some unknown error!........")
def openimage():
    clear_image_and_listbox()
    global label_paths
    global folder_path
    global image_paths
    folder_path = filedialog.askdirectory()
    if folder_path:
        label_paths.clear()
        image_paths = []
        image_listbox.delete(0, "end") 
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder_path, filename))
                image_listbox.insert("end", filename)
        dynamic_text.set(f"No of images : {len(image_paths)}")
        if image_listbox.size() == 0:
            dynamic_text.set("there are no images")
            tk.messagebox.showinfo("Alert","There are no images in the selected folder")
        if image_paths:
            show_first_image()

def show_first_image():
    global label_paths
    global image_paths
    if len(label_paths) == 0:
        if image_paths:
            selected_image_path = image_paths[0]
            image = Image.open(selected_image_path)
            image = image.resize((1200,700))
            photo = ImageTk.PhotoImage(image)
            image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            image_canvas.bg_image = photo
            # image_label.config(image=photo)
            # image_label.image = photo
        else:
            # If there are no images, display a blank image
            blank_image = Image.new("RGB", (1200, 700), "white")
            blank_photo = ImageTk.PhotoImage(blank_image)
            image_canvas.create_image(0, 0, anchor=tk.NW, image=blank_photo)
            image_canvas.bg_image = blank_photo
            # image_label.config(image=blank_photo)
            # image_label.image = blank_photo
    else:
        if image_paths:
            selected_image_path = image_paths[0]
            show_rectangle(selected_image_path,label_paths[0])
        else:
            blank_image = Image.new("RGB", (1200, 700), "white")
            blank_photo = ImageTk.PhotoImage(blank_image)
            image_canvas.create_image(0, 0, anchor=tk.NW, image=blank_photo)
            image_canvas.bg_image = blank_photo
            # image_label.config(image=blank_photo)
            # image_label.image = blank_photo


root = tk.Tk()
screen_width =  root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")
root.title('Autolabeler')

# middle section
middle_frame = tk.Frame(root)
middle_frame.configure(bg="whitesmoke")
middle_frame.pack_propagate(False)
middle_frame.pack(fill='both',expand=True) 
# image_label = tk.Label(middle_frame)
# image_label.place(x=120,y=50)

# canvas to show rectangles 
image_canvas = tk.Canvas(middle_frame, width=1200, height=700)
image_canvas.place(x=120, y=50)

# left section
left_frame = tk.Frame(root, width="100")
left_frame.pack_propagate(False)
left_frame.place(x=0, y=0, relheight=1)  
left_frame.configure(bg='lightgray')

# open button
a = Image.open("F:\\dsk\\open-folder.png")
a=a.resize((40,40))
file_png = ImageTk.PhotoImage(a)
file_button = tk.Button(left_frame, image=file_png, command=openimage)
file_button.pack(pady=20)

# option menu
option_var = tk.StringVar(root)
options = ["head", "gun", "person"]
option_menu = ttk.OptionMenu(left_frame, option_var, options[0],*options)
option_menu.pack(pady=20)


# startbutton
b = Image.open("F:\\dsk\\play.png")
b=b.resize((40,40))
file_png2 = ImageTk.PhotoImage(b)
start_button = tk.Button(left_frame, image=file_png2, command= run_second_code)
start_button.pack(pady=20)


# right section
right_frame = tk.Frame(root, width="250")
right_frame.pack_propagate(False)
right_frame.place(relx=1, x=-250, y=0, relheight=1)
right_frame.configure(bg="lightgray")

# section one
dynamic_text = tk.StringVar()
dynamic_text.set("Initial Text")
sec_one = tk.Label(right_frame,textvariable=dynamic_text,height=10)
sec_one.pack(fill="x")

# scrollbar and listbox
scrollbar = ttk.Scrollbar(right_frame, orient="vertical")
image_listbox = tk.Listbox(right_frame, yscrollcommand=scrollbar.set)
scrollbar.config(command=image_listbox.yview)
scrollbar.pack(side="right", fill="y")
image_listbox.pack(side="right",fill="both",expand=True)
image_listbox.bind("<<ListboxSelect>>", update_image)

root.mainloop()


