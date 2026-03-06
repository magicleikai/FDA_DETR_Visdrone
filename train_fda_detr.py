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
        epochs=300,  # 👈 把上限拉高到 300，给 Transformer 充分的拟合时间
        patience=30,  # 👈 黄金参数：如果连续 30 个 Epoch mAP50 没有突破，就自动停止训练！绝不浪费算力
        imgsz=640,
        batch=4,
        device="0",
        workers=8,
        project="FDA_DETR_Runs",
        name="visdrone_rtdetr_L",
        deterministic=False,
        mosaic=1.0,
        mixup=0.1,
        optimizer="AdamW",
        lr0=0.0001,
        weight_decay=0.0001,
        amp=True,
        save_period=10
    )

    print("[SUCCESS] 训练任务结束！")


if __name__ == '__main__':
    main()