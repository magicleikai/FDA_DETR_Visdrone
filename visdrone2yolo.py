import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_visdrone_to_yolo(visdrone_dir: str, yolo_dir: str):
    """
    将 VisDrone 数据集转换为完全适配 Ultralytics 的 YOLO 格式。
    包含完整的类别映射、脏数据过滤以及边界框归一化逻辑。
    """
    visdrone_path = Path(visdrone_dir)
    yolo_path = Path(yolo_dir)

    # VisDrone 包含的三个核心子集 (忽略 test-challenge，因为通常没有开放真实标签)
    splits = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']
    yolo_splits = ['train', 'val', 'test']

    # VisDrone 类别映射 (根据官方评测标准，忽略 0 和 11)
    # 原始类别: 0:ignored, 1:pedestrian, 2:people, 3:bicycle, 4:car, 5:van,
    #          6:truck, 7:tricycle, 8:awning-tricycle, 9:bus, 10:motor, 11:others
    category_mapping = {
        1: 0,  # pedestrian
        2: 1,  # people
        3: 2,  # bicycle
        4: 3,  # car
        5: 4,  # van
        6: 5,  # truck
        7: 6,  # tricycle
        8: 7,  # awning-tricycle
        9: 8,  # bus
        10: 9  # motor
    }

    for split, yolo_split in zip(splits, yolo_splits):
        print(f"\n[INFO] 正在处理数据子集: {split} -> {yolo_split}")

        # 原始数据路径
        images_dir = visdrone_path / split / 'images'
        annotations_dir = visdrone_path / split / 'annotations'

        if not images_dir.exists() or not annotations_dir.exists():
            print(f"[WARNING] 找不到路径 {images_dir}，跳过该子集。")
            continue

        # 目标数据路径创建
        out_images_dir = yolo_path / 'images' / yolo_split
        out_labels_dir = yolo_path / 'labels' / yolo_split
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        # 遍历所有标注文件
        for label_file in tqdm(list(annotations_dir.glob('*.txt')), desc=f"Converting {yolo_split}"):
            image_file = images_dir / (label_file.stem + '.jpg')
            out_label_file = out_labels_dir / label_file.name
            out_image_symlink = out_images_dir / image_file.name

            if not image_file.exists():
                continue

            # 读取图像尺寸以进行归一化
            img = cv2.imread(str(image_file))
            if img is None:
                continue
            height, width, _ = img.shape

            # 处理标注文件
            valid_labels = []
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split(',')
                    if len(data) < 8:
                        continue

                    bbox_left = float(data[0])
                    bbox_top = float(data[1])
                    bbox_width = float(data[2])
                    bbox_height = float(data[3])
                    score = int(data[4])
                    category_id = int(data[5])

                    # 核心过滤逻辑：过滤掉被判定为0分的无效标注，以及 ignored(0) 和 others(11)
                    if score == 0 or category_id not in category_mapping:
                        continue

                    # 将坐标转换为中心点坐标并归一化到 [0, 1]
                    x_center = (bbox_left + bbox_width / 2.0) / width
                    y_center = (bbox_top + bbox_height / 2.0) / height
                    w_norm = bbox_width / width
                    h_norm = bbox_height / height

                    # 越界修正 (极少数脏数据边界框可能超出图像)
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))

                    yolo_class_id = category_mapping[category_id]
                    valid_labels.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            # 只有当该图片存在有效目标时，才保存标签并复制/链接图片
            if valid_labels:
                with open(out_label_file, 'w', encoding='utf-8') as f:
                    f.writelines(valid_labels)

                if not out_image_symlink.exists():
                    try:
                        # 尝试创建硬链接以节省庞大的硬盘空间
                        os.link(str(image_file.absolute()), str(out_image_symlink.absolute()))
                    except OSError:
                        # 如果系统不支持(如跨盘符)，则退化为物理复制
                        shutil.copy(str(image_file), str(out_image_symlink))

    print("\n[SUCCESS] VisDrone 数据集已成功转换为 YOLO 格式！")


if __name__ == "__main__":
    # 完美匹配你截图中的路径结构
    VISDRONE_RAW_DIR = r"D:\pythonProjects\FDA_DETR_Visdrone\dataset\VisDrone2019-DET"
    # 将清洗后的数据输出到同级目录下的 VisDrone_YOLO 文件夹
    YOLO_OUT_DIR = r"D:\pythonProjects\FDA_DETR_Visdrone\dataset\VisDrone_YOLO"

    convert_visdrone_to_yolo(VISDRONE_RAW_DIR, YOLO_OUT_DIR)