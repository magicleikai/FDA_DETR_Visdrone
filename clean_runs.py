import os
import shutil


def clean_workspace():
    print("🧹 开始执行无情清场...")

    targets_to_delete = [
        "runs/detect/FDA_DETR_Runs",
        "wandb"
    ]

    for target in targets_to_delete:
        if os.path.exists(target):
            try:
                shutil.rmtree(target)
                print(f"✅ 已强行抹除: {target}")
            except Exception as e:
                print(f"❌ 删除 {target} 失败: {e}")
        else:
            print(f"➖ {target} (不存在，跳过)")

    print("\n✨ 物理清理完成！现在的环境绝对纯净。")
    print("🚀 赶紧敲下 `python run_autodl.py`，让 AdamW 接管战场吧！")


if __name__ == "__main__":
    clean_workspace()