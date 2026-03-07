import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import RTDETR


class FeatureExtractor:
    """提取指定网络层的特征图用于可视化"""

    def __init__(self, model, layer_name):
        self.features = None
        self.hook = None
        # 遍历底层网络结构，找到我们的 WaveletFPN 模块并挂上钩子
        for name, module in model.model.named_modules():
            # 通过类名匹配我们的频域解耦层
            if layer_name in str(type(module)):
                print(f"[INFO] 成功挂载 Hook 到层: {name} ({layer_name})")
                self.hook = module.register_forward_hook(self.hook_fn)
                break

        if self.hook is None:
            raise ValueError(f"未能找到包含 '{layer_name}' 的网络层！")

    def hook_fn(self, module, input, output):
        # 抓取前向传播时的输出张量并保存
        self.features = output.detach().cpu()

    def remove(self):
        if self.hook:
            self.hook.remove()


def generate_wavelet_heatmap(weight_path: str, image_path: str, output_path: str):
    print(f"[INFO] 加载模型: {weight_path}")
    # 实例化模型
    model_wrapper = RTDETR(weight_path)

    # 实例化特征提取器，目标锁定我们写的 WaveletFPN_Downsample 层
    extractor = FeatureExtractor(model_wrapper.model, "WaveletFPN_Downsample")

    # 读取原始图像用于后续的热力图叠加
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    h_ori, w_ori = img_bgr.shape[:2]

    print(f"[INFO] 正在对图像进行前向推理并捕获高频特征...")
    # 执行一次推理，触发 Hook 抓取特征
    _ = model_wrapper.predict(image_path, imgsz=640, verbose=False)

    if extractor.features is not None:
        # features shape: [Batch=1, Channels, H, W]
        # 我们将所有通道的激活值进行求平均，得到二维的整体注意力图
        activation_map = torch.mean(extractor.features[0], dim=0).numpy()

        # 对激活图进行极值归一化到 [0, 1]
        activation_map = np.maximum(activation_map, 0)
        activation_map /= (np.max(activation_map) + 1e-8)

        # 将激活图放大到与原始输入图像一样的尺寸
        activation_map_resized = cv2.resize(activation_map, (w_ori, h_ori))

        # 将 [0,1] 的浮点数转化为 [0, 255] 的 uint8 灰度图
        heatmap = np.uint8(255 * activation_map_resized)

        # 应用伪彩色映射 (COLORMAP_JET: 越关注的地方越红，不关注的地方越蓝)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 将热力图与原图按比例混合 (例如 40% 的热力图 + 60% 的原图)
        superimposed_img = cv2.addWeighted(heatmap_color, 0.4, img_bgr, 0.6, 0)

        # 保存神图
        cv2.imwrite(output_path, superimposed_img)
        print(f"[SUCCESS] 高频注意力热力图生成成功: {output_path}")
    else:
        print("[ERROR] 未能成功抓取到特征图。")

    # 释放钩子
    extractor.remove()


if __name__ == "__main__":
    # 等 4090 训练完，换上你的 best.pt
    WEIGHT_PATH = r"../runs/detect/FDA_DETR_Runs/visdrone_rtdetr_L_final/weights/best.pt"
    # 选一张最具代表性的、包含大量微小物体的图
    IMAGE_PATH = r"../dataset/VisDrone_YOLO/images/val/0000001_02999_d_0000001.jpg"
    OUTPUT_PATH = r"../paper_figures/heatmap/wavelet_attention_heatmap.jpg"

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    generate_wavelet_heatmap(WEIGHT_PATH, IMAGE_PATH, OUTPUT_PATH)