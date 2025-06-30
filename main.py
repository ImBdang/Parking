from ultralytics import solutions
import ParkingHandle 
import cv2
import os
from tqdm import tqdm
from sahi import AutoDetectionModel


def run(model_name, data_num, sahi=False):

    data_name = "data" + data_num

    MODEL_DIR = f"models/{model_name}.pt"
    DATA_DIR = f"data/{data_name}"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=f"models/{model_name}.pt", 
        confidence_threshold=0.15,
        device="cpu"
    )

    cap = cv2.VideoCapture(f"{DATA_DIR}/video.mp4")
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(f"{'SAHI_' if sahi else ''}parking_{model_name}.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    vid_name = f"{'SAHI_' if sahi else ''}parking_{model_name}.avi"

    parkingmanager = ParkingHandle.ParkingManagement(        
        model=MODEL_DIR,  
        json_file=f"{DATA_DIR}/bounding_boxes.json",
        sahi_detection_model=detection_model, 
        sahi=sahi
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
        while cap.isOpened():
            ret, im0 = cap.read()
            if not ret:
                pbar.update(total_frames - pbar.n)
                break
            hsv_im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_im0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_v = clahe.apply(v)
            enhanced_hsv_im0 = cv2.merge([h, s, enhanced_v])
            im0_processed = cv2.cvtColor(enhanced_hsv_im0, cv2.COLOR_HSV2BGR)
            # im0_processed = cv2.cvtColor(enhanced_hsv_im0, cv2.COLOR_GRAY2BGR)
            results = parkingmanager(im0_processed)
            # print(results)  # access the output
            video_writer.write(results.plot_im)  
            pbar.update(1) 

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  
    os.system(f"mv {vid_name} predict_videos/{data_name}/")

    print("DA CHAY XONG")

def mark_point():
    solutions.ParkingPtsSelection()



# mark_point()

run("cars", "7", False)