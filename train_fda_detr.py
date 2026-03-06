from ultralytics import RTDETR


def main():
    print("[INFO] 正在拉取官方 rtdetr-l.pt 权重...")
    # 这行代码会成功触发官方下载器下载 HGNetv2 权重
    _ = RTDETR("rtdetr-l.pt")

    print("[INFO] 正在初始化 FDA-DETR (HGNetv2-L) 模型架构...")
    # 在内存中构建我们刚写的 L 版本专属网络拓扑
    model = RTDETR("fda_rtdetr_l.yaml")

    print("[INFO] 将官方权重注入 FDA-DETR 底层...")
    # 完美继承 Backbone 和未被魔改层的几十万小时训练结晶！
    model.load("rtdetr-l.pt")

    print("[INFO] 开始训练流程...")
    results = model.train(
        data="visdrone_fda.yaml",
        epochs=100,  # 既然有了预训练权重，100轮足够起飞
        imgsz=640,
        batch=4,  # L 版本模型较大，显存如果不够可以改成 2
        device="0",
        workers=8,
        project="FDA_DETR_Runs",
        name="visdrone_rtdetr_L",
        deterministic=False,

        # 针对极小目标的数据增强
        mosaic=1.0,
        mixup=0.1,

        # 优化器
        optimizer="AdamW",
        lr0=0.0001,
        weight_decay=0.0001,
        amp=True,  # 混合精度加速
        save_period=10
    )

    print("[SUCCESS] 训练任务结束！")


if __name__ == '__main__':
    main()