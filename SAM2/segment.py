from ultralytics import SAM
import os


picture_name = "frame1"
data_name = "data2"
target = "train"

DATA_DIR = f"data/{data_name}"


# Load a model
model = SAM("../models/sam2.1_b.pt")
with open(f"points.txt", "r") as f:
    content = f.read()

points = eval(content)


# Run inference with multiple point prompts (Provide the points coordinates for
# person area, ensuring that only the person is segmented in the entire image)
results = model(f"../{DATA_DIR}/{picture_name}.jpg",
                points=points)

results[0].show(labels=False)  # Display results

for i, res in enumerate(results):
    normal_bboxs = res.boxes.xywhn
    with open(f"../dataset/handmade/labels/{target}/{data_name}_{picture_name}.txt", "w", encoding="utf-8") as f:
        for nbbox in normal_bboxs:
            x, y, w, h = nbbox
            f.write(f"0 {x} {y} {w} {h}\n")

os.system(f"cp ../{DATA_DIR}/{picture_name}.jpg ../dataset/handmade/images/{target}/{data_name}_{picture_name}.jpg")