import os
import traceback
from ultralytics import RTDETR


def main():
    # 指定训练输出的保存目录和权重路径
    project_dir = "FDA_DETR_Runs"
    run_name = "visdrone_rtdetr_L"
    resume_weights = f"{project_dir}/{run_name}/weights/last.pt"

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
            workers=16,  # 👈 适当增加多线程加载
            cache=True,  # 👈 终极杀招：把数据集全部载入内存
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
        # 保底执行模块：无论上面是顺利跑完 100 轮，还是中间报错崩溃，都会走到这一步！
        print("[INFO] 💤 训练进程已结束，正在触发系统强制关机以停止计费...")
        # os.system("shutdown")