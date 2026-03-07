import os
from pathlib import Path
from ultralytics import RTDETR


def generate_paper_comparison_images(weight_path: str, source_dir: str, output_dir: str):
    """
    加载训练好的 FDA-DETR 权重，对图像进行推理，生成用于论文对比的高清检测图。
    """
    print(f"[INFO] 加载 FDA-DETR 模型权重: {weight_path}")
    # 加载你在 4090 上训练得到的 best.pt
    model = RTDETR(weight_path)

    print(f"[INFO] 开始处理目标文件夹: {source_dir}")

    # 针对极小目标的特调推理参数
    results = model.predict(
        source=source_dir,
        conf=0.15,  # 置信度阈值：VisDrone 目标极小，适当调低置信度可以展现极强的召回率 (Recall)
        iou=0.45,  # NMS 阈值：密集场景下，调高 IoU 阈值防止拥挤目标被错误抑制
        imgsz=640,  # 保持与训练一致的分辨率，或者如果显存够大可提升至 800-1024 进一步压榨性能
        save=True,  # 自动保存画好框的图片
        line_width=1,  # 极其关键：小目标场景下画框的线一定要细！否则红线会把目标完全遮挡
        show_labels=False,  # 论文对比图中通常不需要显示杂乱的类别文字，只显示框即可
        show_conf=False,  # 隐藏置信度分数，保持画面整洁
        project=output_dir,  # 保存的主目录
        name="high_res_preds"  # 保存的子目录名
    )

    print(f"[SUCCESS] 所有对比图已生成，保存在: {os.path.join(output_dir, 'high_res_preds')}")


if __name__ == "__main__":
    # 等你在 4090 上跑完后，把这里的路径换成你的 best.pt 路径
    WEIGHT_PATH = r"../runs/detect/FDA_DETR_Runs/visdrone_rtdetr_L_final/weights/best.pt"
    # 找几张特别密集的验证集/测试集图片放在一个单独的文件夹里，专门用来跑出图
    SOURCE_IMAGES = r"../dataset/VisDrone_YOLO/images/val"
    # 输出结果存放位置
    OUTPUT_DIR = r"../paper_figures/comparison"

    # 确保文件夹存在
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    generate_paper_comparison_images(WEIGHT_PATH, SOURCE_IMAGES, OUTPUT_DIR)