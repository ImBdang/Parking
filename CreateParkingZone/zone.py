import json
with open("data.txt", "r") as f:
    data = f.read()

data = json.loads(data)
data = data["predictions"]

result = []

for item in data:
    center_x = item["x"]
    center_y = item["y"]
    width = item["width"]
    height = item["height"]

    half_width = width / 2
    half_height = height / 2

    top_left = [center_x - half_width, center_y - half_height]
    top_right = [center_x + half_width, center_y - half_height]
    bottom_right = [center_x + half_width, center_y + half_height]
    bottom_left = [center_x - half_width, center_y + half_height]
    
    kq = {
        "points": [top_left, top_right, bottom_right, bottom_left]
    }
    result.append(kq)

with open("bounding_boxes.json", "w") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)