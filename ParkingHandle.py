# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from typing import Any, List, Tuple

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_imshow


class ParkingPtsSelection:
    """
    A class for selecting and managing parking zone points on images using a Tkinter-based UI.

    This class provides functionality to upload an image, select points to define parking zones, and save the
    selected points to a JSON file. It uses Tkinter for the graphical user interface.

    Attributes:
        tk (module): The Tkinter module for GUI operations.
        filedialog (module): Tkinter's filedialog module for file selection operations.
        messagebox (module): Tkinter's messagebox module for displaying message boxes.
        master (tk.Tk): The main Tkinter window.
        canvas (tk.Canvas): The canvas widget for displaying the image and drawing bounding boxes.
        image (PIL.Image.Image): The uploaded image.
        canvas_image (ImageTk.PhotoImage): The image displayed on the canvas.
        rg_data (List[List[Tuple[int, int]]]): List of bounding boxes, each defined by 4 points.
        current_box (List[Tuple[int, int]]): Temporary storage for the points of the current bounding box.
        imgw (int): Original width of the uploaded image.
        imgh (int): Original height of the uploaded image.
        canvas_max_width (int): Maximum width of the canvas.
        canvas_max_height (int): Maximum height of the canvas.

    Methods:
        initialize_properties: Initialize properties for image, canvas, bounding boxes, and dimensions.
        upload_image: Upload and display an image on the canvas, resizing it to fit within specified dimensions.
        on_canvas_click: Handle mouse clicks to add points for bounding boxes on the canvas.
        draw_box: Draw a bounding box on the canvas using the provided coordinates.
        remove_last_bounding_box: Remove the last bounding box from the list and redraw the canvas.
        redraw_canvas: Redraw the canvas with the image and all bounding boxes.
        save_to_json: Save the selected parking zone points to a JSON file with scaled coordinates.

    Examples:
        >>> parking_selector = ParkingPtsSelection()
        >>> # Use the GUI to upload an image, select parking zones, and save the data
    """

    def __init__(self):
        """Initialize the ParkingPtsSelection class, setting up UI and properties for parking zone point selection."""
        try:  # Check if tkinter is installed
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError:  # Display error with recommendations
            import platform

            install_cmd = {
                "Linux": "sudo apt install python3-tk (Debian/Ubuntu) | sudo dnf install python3-tkinter (Fedora) | "
                "sudo pacman -S tk (Arch)",
                "Windows": "reinstall Python and enable the checkbox `tcl/tk and IDLE` on **Optional Features** during installation",
                "Darwin": "reinstall Python from https://www.python.org/downloads/macos/ or `brew install python-tk`",
            }.get(platform.system(), "Unknown OS. Check your Python installation.")

            LOGGER.warning(f" Tkinter is not configured or supported. Potential fix: {install_cmd}")
            return

        if not check_imshow(warn=True):
            return

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.master = self.tk.Tk()  # Reference to the main application window
        self.master.title("Ultralytics Parking Zones Points Selector")
        self.master.resizable(False, False)

        self.canvas = self.tk.Canvas(self.master, bg="white")  # Canvas widget for displaying images
        self.canvas.pack(side=self.tk.BOTTOM)

        self.image = None  # Variable to store the loaded image
        self.canvas_image = None  # Reference to the image displayed on the canvas
        self.canvas_max_width = None  # Maximum allowed width for the canvas
        self.canvas_max_height = None  # Maximum allowed height for the canvas
        self.rg_data = None  # Data for region annotation management
        self.current_box = None  # Stores the currently selected bounding box
        self.imgh = None  # Height of the current image
        self.imgw = None  # Width of the current image

        # Button frame with buttons
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

        self.initialize_properties()
        self.master.mainloop()

    def initialize_properties(self):
        """Initialize properties for image, canvas, bounding boxes, and dimensions."""
        self.image = self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 1280, 720

    def upload_image(self):
        """Upload and display an image on the canvas, resizing it to fit within specified dimensions."""
        from PIL import Image, ImageTk  # Scoped import because ImageTk requires tkinter package

        self.image = Image.open(self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")]))
        if not self.image:
            return

        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height)))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def on_canvas_click(self, event):
        """Handle mouse clicks to add points for bounding boxes on the canvas."""
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box: List[Tuple[int, int]]):
        """Draw a bounding box on the canvas using the provided coordinates."""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self):
        """Remove the last bounding box from the list and redraw the canvas."""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self):
        """Redraw the canvas with the image and all bounding boxes."""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self):
        """Save the selected parking zone points to a JSON file with scaled coordinates."""
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]

        from io import StringIO  # Function level import, as it's only required to store coordinates

        write_buffer = StringIO()
        json.dump(data, write_buffer, indent=4)
        with open("bounding_boxes.json", "w", encoding="utf-8") as f:
            f.write(write_buffer.getvalue())
        self.messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")


class ParkingManagement(BaseSolution):
    """
    Manages parking occupancy and availability using YOLO model for real-time monitoring and visualization.

    This class extends BaseSolution to provide functionality for parking lot management, including detection of
    occupied spaces, visualization of parking regions, and display of occupancy statistics.

    Attributes:
        json_file (str): Path to the JSON file containing parking region details.
        json (List[Dict]): Loaded JSON data containing parking region information.
        pr_info (Dict[str, int]): Dictionary storing parking information (Occupancy and Available spaces).
        arc (Tuple[int, int, int]): RGB color tuple for available region visualization.
        occ (Tuple[int, int, int]): RGB color tuple for occupied region visualization.
        dc (Tuple[int, int, int]): RGB color tuple for centroid visualization of detected objects.

    Methods:
        process: Process the input image for parking lot management and visualization.

    Examples:
        >>> from ultralytics.solutions import ParkingManagement
        >>> parking_manager = ParkingManagement(model="yolo11n.pt", json_file="parking_regions.json")
        >>> print(f"Occupied spaces: {parking_manager.pr_info['Occupancy']}")
        >>> print(f"Available spaces: {parking_manager.pr_info['Available']}")
    """

    def __init__(self, sahi_detection_model, sahi,**kwargs: Any):
        """Initialize the parking management system with a YOLO model and visualization settings."""
        super().__init__(**kwargs)
        self.sahi = sahi
        self.json_file = self.CFG["json_file"]  # Load parking regions JSON data
        if self.json_file is None:
            LOGGER.warning("json_file argument missing. Parking region details required.")
            raise ValueError("❌ Json file path can not be empty")

        with open(self.json_file) as f:
            self.json = json.load(f)

        
        self.sahi_model = sahi_detection_model
    
        self.pr_info = {"Cho da dung": 0, "Cho con trong": 0}  # Dictionary for parking information

        self.arc = (0, 0, 255)  # Available region color
        self.occ = (0, 255, 0)  # Occupied region color
        self.dc = (255, 0, 189)  # Centroid color for each box

    def process(self, im0: np.ndarray) -> SolutionResults:
        """
        Process the input image for parking lot management and visualization.

        This function analyzes the input image, extracts tracks, and determines the occupancy status of parking
        regions defined in the JSON file. It annotates the image with occupied and available parking spots,
        and updates the parking information.

        Args:
            im0 (np.ndarray): The input inference image.

        Returns:
            (SolutionResults): Contains processed image `plot_im`, 'filled_slots' (number of occupied parking slots),
                'available_slots' (number of available parking slots), and 'total_tracks' (total number of tracked objects).

        Examples:
            >>> parking_manager = ParkingManagement(json_file="parking_regions.json")
            >>> image = cv2.imread("parking_lot.jpg")
            >>> results = parking_manager.process(image)
        """
        # self.extract_tracks(im0)  # Extract tracks from im0

        if self.sahi:
            #========================= BDANG ===============================
            from sahi.predict import get_sliced_prediction
            sahi_result = get_sliced_prediction(im0, self.sahi_model, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
            sahi_result = sahi_result.object_prediction_list
            es, fs = len(self.json), 0  # Empty slots, filled slots
            annotator = SolutionAnnotator(im0, self.line_width)  # Initialize annotator

            for region in self.json:
                # Convert points to a NumPy array with the correct dtype and reshape properly
                pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
                rg_occupied = False  # Occupied region initialization

                for item in sahi_result:
                    x_min, y_min, x_max, y_max = item.bbox.to_xyxy()
                    xc = int((x_min + x_max) / 2)
                    yc = int((y_min + y_max) / 2)

                    dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                    #kiem tra doi tuong vao tam do xe chua
                    if dist >= 0:
                        rg_occupied = True
                        break
                fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
                # Plot regions
                cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=2)


        else:
            self.extract_tracks(im0)  # Extract tracks from im0
            es, fs = len(self.json), 0  # Empty slots, filled slots
            annotator = SolutionAnnotator(im0, self.line_width)  # Initialize annotator
            i = 0
            for region in self.json:
                # Convert points to a NumPy array with the correct dtype and reshape properly
                pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
                rg_occupied = False  # Occupied region initialization
                self.capnhat(rg_occupied, i)
                for box, cls in zip(self.boxes, self.clss): #Duyet qua list car duoc du doan
                    xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2) #Xac dinh center của bounding box
                    dist = cv2.pointPolygonTest(pts_array, (xc, yc), False) #Su dung ham co san cua opencv de so sanh 
                    if dist >= 0: # Gia tri tra ve >=0 la truong hop ma bouding bõx car nam trong parking slot
                        rg_occupied = True # Danh dau parking slot da duoc su dung
                        self.capnhat(rg_occupied, i)
                        break
                    
                fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
                # Plot regions
                cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=2)
                i = i + 1

            self.pr_info["Cho da dung"], self.pr_info["Cho con trong"] = fs, es

            annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)

            plot_im = annotator.result()
            self.display_output(plot_im)  # Display output with base class function



        self.pr_info["Cho da dung"], self.pr_info["Cho con trong"] = fs, es
        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            filled_slots=self.pr_info["Cho da dung"],
            available_slots=self.pr_info["Cho con trong"],
            total_tracks=len(self.track_ids),
        )
    
    def __call__(self, im0: np.ndarray) -> SolutionResults:
        return self.process(im0)
    
    def capnhat(self, status: bool, i: int):
        with open("web/bounding_boxes.json", "r") as f:
            data = json.load(f)

        data[i]['st'] = status  

        with open("web/bounding_boxes.json", "w") as f:
            json.dump(data, f, indent=4)
