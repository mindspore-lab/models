import os


def get_device_id():
    device_id_ascend = int(os.getenv("DEVICE_ID", "0"))
    device_id_gpu = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    return max(device_id_ascend, device_id_gpu)


def get_device_num():
    device_num_ascend = int(os.getenv("RANK_SIZE", "1"))
    device_num_gpu = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    return max(device_num_ascend, device_num_gpu)


def get_local_device_num():
    local_device_num_ascend = min(get_device_num(), 8)
    local_device_num_gpu = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
    return max(local_device_num_ascend, local_device_num_gpu)


def get_rank_id():
    global_rank_id_ascend = int(os.getenv("RANK_ID", "0"))
    global_rank_id_gpu = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    return max(global_rank_id_ascend, global_rank_id_gpu)
