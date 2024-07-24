import concurrent.futures
import moxing as mox


def copy_task(src, dst):
    mox.file.copy_parallel(src, dst)


# 定义源路径和目标路径
src = 's3://bucket-5541/wenzhuoliu/torch_ds/cifar-100-python/'
dst = '/home/ma-user/work/wenzhuoliu/torch_ds/cifar100'
max_works = 64

# 使用 ThreadPoolExecutor 创建线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=max_works) as executor:
    # 提交任务给线程池
    futures = [executor.submit(copy_task, src, dst) for _ in range(max_works)]

    # 等待所有任务完成
    for future in concurrent.futures.as_completed(futures):
        future.result()

print("Copying completed.")
