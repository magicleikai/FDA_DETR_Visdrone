from ultralytics import RTDETR


def main():
    print("[INFO] 正在初始化 FDA-DETR 模型架构...")
    # 1. 加载我们包含 Wavelet-FPN 的定制网络架构配置文件
    model = RTDETR("fda_rtdetr_resnet50.yaml")

    # 可选：如果你有标准 RT-DETR-ResNet50 的预训练权重，可以在这里加载以加速收敛
    # 虽然我们的 Neck 和 Loss 被魔改了，但主干网络(ResNet)的权重是可以完美继承的
    # model.load("rtdetr-resnet50.pt")

    print("[INFO] 开始训练流程...")
    # 2. 启动训练 (配置已针对小目标场景优化)
    results = model.train(
        data="visdrone_fda.yaml",  # 我们刚才写的数据配置文件
        epochs=150,  # VisDrone 数据集较大，建议 100-150 epoch
        imgsz=640,  # 如果显存允许(如 24GB)，可以提升到 800 或 1024 进一步保留高频细节
        batch=4,  # 根据显存大小调整 (RT-DETR显存占用较大，4 或 8 是安全选项)
        device="0",  # 使用 GPU 0
        workers=8,  # DataLoader 线程数
        project="FDA_DETR_Runs",  # 训练结果保存的主目录
        name="visdrone_resnet50",  # 本次实验的名称

        # 针对小目标的特定数据增强参数
        mosaic=1.0,  # 强开 Mosaic，极大地丰富小目标的背景上下文
        mixup=0.1,  # 适度开启 MixUp
        degrees=10.0,  # 轻微旋转

        # 优化器配置
        optimizer="AdamW",  # Transformer 架构首选 AdamW
        lr0=0.0001,  # 初始学习率 (DETR系列要求比 YOLO 更低的学习率)
        weight_decay=0.0001,

        # 保存最优模型
        save_period=10  # 每 10 个 epoch 保存一次检查点
    )

    print("[SUCCESS] 训练任务结束！模型权重已保存在 FDA_DETR_Runs 目录下。")


if __name__ == '__main__':
    main()