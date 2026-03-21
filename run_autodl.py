import os
import traceback
import wandb
from ultralytics import RTDETR

# 优化 PyTorch 底层显存碎片管理，极大降低 OOM 概率
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    # 指定训练输出的保存目录和权重路径
    project_dir = "FDA_DETR_Runs"
    run_name = "visdrone_rtdetr_L"
    # 🚨 核心修复：加上 runs/detect/ 前缀！指向你真正断掉的那个文件夹！
    resume_weights = f"runs/detect/{project_dir}/{run_name}/weights/last.pt"

    # 初始化 W&B，resume="allow" 确保断点续训时曲线无缝拼接
    wandb.init(project=project_dir, name=run_name, resume="allow")

    # 智能分支：判断是接着跑，还是重新跑
    if os.path.exists(resume_weights):
        print(f"[INFO] 发现中断的权重文件 {resume_weights}，正在恢复训练...")
        model = RTDETR(resume_weights)

        # 🚨 核心修复：断点续训时，绝对不能只写 resume=True！
        # 如果只写 resume=True，它会读取 last.pt 里的旧 batch=8 配置，马上又会 OOM。
        # 必须显式传入新的 batch size 来覆盖旧配置，强行度过密集目标难关！
        model.train(
            resume=True,
            data="visdrone_fda.yaml",
            epochs=100,
            imgsz=1024,
            batch=6,  # 👈 强制降为 4，预留充足显存打通最难的图片
            workers=6,
            cache=True,
            device=0,
            project=project_dir,
            name=run_name,
        )
    else:
        print("[INFO] 未发现历史记录，开始基于 5090 算力进行全新训练...")
        model = RTDETR("fda_rtdetr_l.yaml")
        model.load("rtdetr-l.pt")

        # 5090 稳健跑法参数配置
        model.train(
            data="visdrone_fda.yaml",
            epochs=100,
            imgsz=1024,
            batch=8,  # 👈 全新训练时也降到 6，给你的高频能量特征图留出 20% 显存缓冲池
            workers=6,
            cache=True,
            device=0,
            project=project_dir,
            name=run_name,
            # 优化器与防烧毁设置
            optimizer='AdamW',
            lr0=0.0001,
            weight_decay=0.0001,
            warmup_epochs=0.0,  # 彻底关闭狂暴预热
            warmup_bias_lr=0.0001  # 彻底关闭偏置项预热
        )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # 异常捕获模块：如果半夜 OOM 或者代码报错，记录下完整的报错堆栈
        print("\n=========================================================")
        print("[ERROR] 🚨 训练过程中发生崩溃！详细错误堆栈如下：")
        traceback.print_exc()
        print("=========================================================\n")
    finally:
        # 致命防坑！确保在系统关机前，把最后的 Loss 和指标无损上传到云端
        if wandb.run is not None:
            print("[INFO] 正在同步最后的数据到 Weights & Biases 云端...")
            wandb.finish()

        # 保底执行模块：无论顺利跑完还是报错崩溃，一定触发关机保护钱包！
        print("[INFO] 💤 训练进程已结束，正在触发系统强制关机以停止计费...")
        # 如果你确认代码没问题准备过夜挂机，把下面这行前面的井号 (#) 删掉
        # os.system("shutdown")