import cv2
import os
import albumentations as A
from tqdm import tqdm 


INPUT_IMAGES_DIR = '../datatrain/images/train' 
INPUT_LABELS_DIR = '../datatrain/labels/train' 

OUTPUT_IMAGES_DIR = 'augmented_dataset/images/train'
OUTPUT_LABELS_DIR = 'augmented_dataset/labels/train' 

NUM_AUGMENTATIONS_PER_IMAGE = 3 

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

transform = A.Compose([
    # Biến đổi hình học
    A.HorizontalFlip(p=0.5), # Lật ngang với xác suất 50%
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.6), # Dịch chuyển, scale, xoay
    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5), # Cắt và thay đổi kích thước ngẫu nhiên

    # Biến đổi màu sắc
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6), # Thay đổi độ sáng/tương phản
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.6), # Thay đổi kênh màu RGB
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6), # Thay đổi Hue, Saturation, Value

    # Thêm nhiễu hoặc che khuất
    A.GaussNoise(p=0.3), # Thêm nhiễu Gaussian
    A.CoarseDropout(max_holes=8, max_h_size=64, max_w_size=64, p=0.3), # Che khuất một phần ngẫu nhiên (tương tự Cutout)

    # Đảm bảo bounding box nằm trong ảnh sau khi biến đổi
    A.LongestMaxSize(max_size=640, p=1.0), # Đảm bảo ảnh có kích thước tối đa 640 ở cạnh dài nhất
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=1.0) # Đệm ảnh để có kích thước 640x640
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])) # Rất quan trọng: format='yolo'

# --- Hàm đọc nhãn YOLO ---
def read_yolo_labels(label_path, img_width, img_height):
    bboxes = []
    class_ids = []
    if not os.path.exists(label_path):
        return bboxes, class_ids

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            center_x, center_y, width, height = parts[1:]
            
            # Albumentations với định dạng 'yolo' mong đợi:
            # [x_center, y_center, width, height, class_id] (đã chuẩn hóa)
            # Chúng ta sẽ trả về mảng 2D cho bboxes và mảng 1D cho class_ids
            bboxes.append([center_x, center_y, width, height])
            class_ids.append(class_id)
    return bboxes, class_ids

# --- Hàm lưu nhãn YOLO ---
def save_yolo_labels(label_path, bboxes, class_ids):
    with open(label_path, 'w') as f:
        for i, bbox in enumerate(bboxes):
            # Kiểm tra xem bounding box có hợp lệ không (có thể bị loại bỏ sau augmentation)
            if len(bbox) == 4: # Chỉ lấy 4 giá trị center_x, center_y, width, height
                f.write(f"{class_ids[i]} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

# --- Xử lý Augmentation ---
image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

print(f"Bắt đầu tăng cường dữ liệu cho {len(image_files)} ảnh gốc...")

for img_file in tqdm(image_files):
    img_name_without_ext = os.path.splitext(img_file)[0]
    img_path = os.path.join(INPUT_IMAGES_DIR, img_file)
    label_path = os.path.join(INPUT_LABELS_DIR, img_name_without_ext + '.txt')

    # Đọc ảnh và nhãn
    image = cv2.imread(img_path)
    if image is None:
        print(f"Không đọc được ảnh: {img_path}. Bỏ qua.")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển BGR sang RGB (Albumentations thường làm việc với RGB)

    # Lấy kích thước ảnh gốc để đọc nhãn chuẩn hóa
    h_orig, w_orig, _ = image.shape
    
    original_bboxes, original_class_ids = read_yolo_labels(label_path, w_orig, h_orig)

    # Lưu ảnh và nhãn gốc vào thư mục mới (nếu bạn muốn giữ chúng trong tập augmented)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, img_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    save_yolo_labels(os.path.join(OUTPUT_LABELS_DIR, img_name_without_ext + '.txt'), original_bboxes, original_class_ids)

    # Tạo các phiên bản augmented
    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
        try:
            # Áp dụng transform
            transformed = transform(image=image, bboxes=original_bboxes, class_ids=original_class_ids)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_ids = transformed['class_ids'] # Đảm bảo lấy lại class_ids từ output

            # Tạo tên file mới
            new_img_name = f"{img_name_without_ext}_aug_{i}.jpg"
            new_label_name = f"{img_name_without_ext}_aug_{i}.txt"

            # Lưu ảnh đã augmented
            cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, new_img_name), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

            # Lưu nhãn đã augmented
            save_yolo_labels(os.path.join(OUTPUT_LABELS_DIR, new_label_name), transformed_bboxes, transformed_class_ids)

        except Exception as e:
            print(f"Lỗi khi tăng cường ảnh {img_file}, lần {i}: {e}. Bỏ qua bản này.")
            continue

print("Hoàn tất quá trình tăng cường dữ liệu!")