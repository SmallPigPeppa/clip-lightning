#!/usr/bin/env python
import torch
import os
import argparse

def save_new_model_weights(checkpoint_path1, checkpoint_path2, output_path, ratio1, ratio2):
    """
    加载来自两个检查点路径的模型权重，根据指定的混合比例进行混合，
    并将新的权重保存到指定的输出路径，键名为 'model'。

    参数：
        checkpoint_path1 (str): 第一个检查点的路径。
        checkpoint_path2 (str): 第二个检查点的路径。
        output_path (str): 保存混合后模型权重的路径。
        ratio1 (float): 第一个模型的混合比例。
        ratio2 (float): 第二个模型的混合比例。
    """
    if checkpoint_path1 and checkpoint_path2:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")

        # 归一化混合比例
        total_ratio = ratio1 + ratio2
        normalized_ratio1 = ratio1 / total_ratio
        normalized_ratio2 = ratio2 / total_ratio

        # 从检查点加载模型参数
        checkpoint1 = torch.load(checkpoint_path1.strip(), map_location=torch.device('cpu'))
        model_params1 = checkpoint1['model']

        checkpoint2 = torch.load(checkpoint_path2.strip(), map_location=torch.device('cpu'))
        model_params2 = checkpoint2['model']

        # 初始化混合参数
        mixed_params = {}
        for k in model_params1.keys():
            mixed_params[k] = (
                model_params1[k] * normalized_ratio1 +
                model_params2[k] * normalized_ratio2
            )

        # 保存混合参数
        torch.save({'model': mixed_params}, output_path)
        print("混合后的模型权重已成功保存。")
    else:
        print("错误：必须提供两个检查点路径。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='根据指定的比例混合两个模型检查点。')
    parser.add_argument('checkpoint_path1', type=str, help='第一个检查点的路径')
    parser.add_argument('checkpoint_path2', type=str, help='第二个检查点的路径')
    parser.add_argument('output_path', type=str, help='保存混合后模型的路径')
    parser.add_argument('ratio1', type=float, help='第一个模型的混合比例')
    parser.add_argument('ratio2', type=float, help='第二个模型的混合比例')

    args = parser.parse_args()

    save_new_model_weights(args.checkpoint_path1, args.checkpoint_path2, args.output_path, args.ratio1, args.ratio2)
