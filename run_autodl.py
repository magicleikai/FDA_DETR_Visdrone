import os
import traceback
import wandb  # 👈 新增：引入 wandb 库
from ultralytics import RTDETR


def main():
    # 指定训练输出的保存目录和权重路径
    project_dir = "FDA_DETR_Runs"
    run_name = "visdrone_rtdetr_L"
    resume_weights = f"{project_dir}/{run_name}/weights/last.pt"

    # 👈 新增：在代码启动时显式初始化 W&B
    # resume="allow" 意味着如果系统检测到这是断点续训，它会自动把曲线拼接到上次的图表后面！
    wandb.init(project=project_dir, name=run_name, resume="allow")

    # 智能分支：判断是接着跑，还是重新跑
    if os.path.exists(resume_weights):
        print(f"[INFO] 发现中断的权重文件 {resume_weights}，正在恢复训练...")
        model = RTDETR(resume_weights)
        # resume=True 会自动接管之前所有的超参数配置，不用再写 batch、imgsz 等
        model.train(resume=True)
    else:
        print("[INFO] 未发现历史记录，开始基于 5090 算力进行全新训练...")
        model = RTDETR("fda_rtdetr_l.yaml")
        model.load("rtdetr-l.pt")

        # 5090 的“榨干级”极限参数配置
        model.train(
            data="visdrone_fda.yaml",
            epochs=100,
            imgsz=800,
            batch=12,
            workers=20,
            cache=True,
            device=0,
            project=project_dir,
            name=run_name
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
        # 👈 新增：致命防坑！确保在系统关机前，把最后的 Loss 和指标上传到云端
        if wandb.run is not None:
            print("[INFO] 正在同步最后的数据到 Weights & Biases 云端...")
            wandb.finish()

        # 保底执行模块：无论上面是顺利跑完 100 轮，还是中间报错崩溃，都会走到这一步！
        print("[INFO] 💤 训练进程已结束，正在触发系统强制关机以停止计费...")
        # os.system("shutdown")  # 跑通之后记得把这行的注释解开