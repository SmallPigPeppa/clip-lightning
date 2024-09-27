import torch
import os


def save_new_model_weights(checkpoint_paths, output_path):
    """
    Load model weights from given checkpoint paths, average them if multiple,
    and save the new weights to the specified output path with 'model' as the key.

    Args:
        checkpoint_paths (str): A single path or multiple paths separated by commas.
        output_path (str): The path where the new model weights will be saved.
    """
    if checkpoint_paths is not None:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        if ',' in checkpoint_paths:
            checkpoint_paths = checkpoint_paths.split(',')
            avg_params = None
            count = 0

            for chkpt_path in checkpoint_paths:
                checkpoint = torch.load(chkpt_path.strip(), map_location=torch.device('cpu'))
                model_params = checkpoint['model']

                if avg_params is None:
                    avg_params = {k: v.clone().detach() for k, v in model_params.items()}
                else:
                    for k in avg_params.keys():
                        avg_params[k] += model_params[k]

                count += 1

            # Compute the average
            for k in avg_params.keys():
                avg_params[k] /= count

            # Save the averaged parameters
            torch.save({'model': avg_params}, output_path)
            print("Averaged model weights saved successfully.")
        else:
            checkpoint = torch.load(checkpoint_paths.strip(), map_location=torch.device('cpu'))
            model_params = checkpoint['model']
            # Save the model parameters
            torch.save({'model': model_params}, output_path)
            print("Model weights saved successfully.")
    else:
        print("No checkpoint paths provided.")


if __name__ == "__main__":
    # 单个检查点路径
    save_new_model_weights('path/to/checkpoint.pth', 'path/to/output/model_weights.pth')

    # 多个检查点路径，用逗号分隔
    save_new_model_weights('path/to/checkpoint1.pth,path/to/checkpoint2.pth', 'path/to/output/model_weights.pth')
