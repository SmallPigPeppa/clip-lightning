import subprocess


def read_base_command(file_path):
    """
    Read the base command from a shell script file and convert it to a dictionary.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    initial_command = lines[0].strip()
    command_dict = {}
    for line in lines[1:]:
        line = line.strip()
        if line.endswith("\\"):
            line = line[:-1].strip()
        key, value = line.split(maxsplit=1)
        command_dict[key] = value

    return initial_command, command_dict


def modify_command_for_task(initial_command, command_dict, task_idx, num_tasks):
    """
    Modify the command dictionary for a specific task index and convert it back to a command string.
    """
    # Update parameters for incremental learning
    command_dict['--data.num_tasks'] = str(num_tasks)
    command_dict['--data.current_task'] = str(task_idx)

    # Update ckpt path
    if task_idx > 0:
        command_dict['--old_ckpt'] = f"task-{task_idx - 1}.ckpt"
    command_dict['--new_ckpt'] = f"task-{task_idx}.ckpt"

    # Update current task id
    command_dict['--data.current_task'] = str(task_idx)

    # Modify the logger name to include the task index
    if '--trainer.logger.name' in command_dict:
        command_dict['--trainer.logger.name'] += f"-task-{task_idx}"

    # Convert dictionary back to command string
    command_parts = [initial_command]
    for key, value in command_dict.items():
        command_parts.append(f"{key} {value} \\")

    # Remove the last backslash
    command_parts[-1] = command_parts[-1].rstrip(" \\")

    modified_command = " \\\n    ".join(command_parts)
    return modified_command


def incremental_learning(num_tasks, script_path):
    """
    Manage the incremental learning process across multiple tasks.
    """
    initial_command, base_command_dict = read_base_command(script_path)
    for task_idx in range(num_tasks):
        print(f"Starting training for task {task_idx}")
        task_specific_command = modify_command_for_task(initial_command, base_command_dict.copy(), task_idx, num_tasks)
        subprocess.run(task_specific_command, shell=True, check=True)
        print(f"Completed training for task {task_idx}")


if __name__ == "__main__":
    num_tasks = 5  # Total number of tasks
    script_path = "train_gpu_r50_aug_bs128_debug_ILR_PLR_DILV2.sh"
    incremental_learning(num_tasks, script_path)
