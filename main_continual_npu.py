import os.path
import subprocess


def read_base_command(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extracting the base command from the first line
    base_command = lines[0].strip().replace(' \\', '')

    # Extracting the parameters
    params_dict = {}
    for line in lines[1:]:
        line = line.strip().replace(' \\', '')
        if line.startswith('--'):
            key, value = line.split(maxsplit=1)
            params_dict[key] = value
    print(params_dict)
    return base_command, params_dict


def modify_command_for_task(base_command, params_dict, task_idx, num_tasks):
    """
    Modify the command dictionary for a specific task index and convert it back to a command string.
    """
    # Update parameters for incremental learning
    params_dict['--data.num_tasks'] = str(num_tasks)
    params_dict['--data.current_task'] = str(task_idx)

    # # Update ckpt path
    if task_idx > 0:
        ckpt_dir = params_dict['--model_checkpoint.dirpath']
        params_dict['--model.old_checkpoint_path'] = os.path.join(ckpt_dir, f"task-{task_idx - 1}.ckpt")
    params_dict['--model_checkpoint.filename'] = f"task-{task_idx}"

    # Update current task id
    params_dict['--data.current_task'] = str(task_idx)

    # Modify the logger name to include the task index
    if '--trainer.logger.name' in params_dict:
        params_dict['--trainer.logger.name'] += f"-task-{task_idx}"

    # Convert dictionary back to command string
    command_parts = [base_command]
    for key, value in params_dict.items():
        command_parts.append(f"{key} {value}")
    modified_command = " \\\n    ".join(command_parts)
    print(modified_command)
    return modified_command


def incremental_learning(num_tasks, script_path):
    """
    Manage the incremental learning process across multiple tasks.
    """
    base_command, params_dict = read_base_command(script_path)
    for task_idx in range(num_tasks):
        print(f"Starting training for task {task_idx}")
        task_specific_command = modify_command_for_task(base_command, params_dict.copy(), task_idx, num_tasks)
        subprocess.run(task_specific_command, shell=True, check=True)
        print(f"Completed training for task {task_idx}")


if __name__ == "__main__":
    num_tasks = 5  # Total number of tasks
    script_path = "train_npu_r50_aug_bs256_debug_ILR_PLR_DILV2.sh"
    incremental_learning(num_tasks, script_path)
