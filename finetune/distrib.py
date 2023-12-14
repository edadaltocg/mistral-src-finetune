import logging
import os

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def setup(backend="nccl"):
    if torch.cuda.is_available():
        backend = "nccl"

    if not dist.is_available():
        logger.warning("Distributed is not available")
        return False

    if os.environ.get("RANK") is None:
        logger.warning(
            """Running without distributed. Try running with `torchrun`.
        Example: 
            torchrun --standalone --nnodes 1 --nproc-per-node 2 genai/distrib.py
        """
        )
        return False

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")
    logger.info(f"rank: {rank}, world_size: {world_size}")

    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}",
    )

    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    if is_dist_avail_and_initialized():
        logger.info(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )
        logger.info(f"Start running distributed code on rank {rank}/{world_size}.")
        logger.info(f"local rank: {local_rank}")

    return True


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    return get_rank() == 0


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def save_model(ddp_model, path):
    if is_main_process():
        torch.save(ddp_model.state_dict(), path)
    if is_dist_avail_and_initialized():
        dist.barrier()


def save_checkpoint(checkpoint, path):
    if is_main_process():
        torch.save(checkpoint, path)
    if is_dist_avail_and_initialized():
        dist.barrier()


def reduce_across_processes(val, rank):
    if not is_dist_avail_and_initialized():
        if isinstance(val, torch.Tensor):
            return val
        return torch.tensor(val)

    if not isinstance(val, torch.Tensor):
        val = torch.tensor(val, device=rank)
    dist.barrier()
    dist.all_reduce(val)
    return val


def demo_basic(*args, **kwargs):
    """
    #SBATCH --nodes=N            # total number of nodes (N to be defined)
    #SBATCH --ntasks-per-node=4  # number of tasks per node (here 4 tasks, or 1 task per GPU)
    #SBATCH --gres=gpu:4         # number of GPUs reserved per node (here 4, or all the GPUs)
    #SBATCH --cpus-per-task=10   # number of cores per task (4x10 = 40 cores, or all the cores)
    export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
    torchrun --standalone --nnodes 1 --nproc-per-node 2 --rdzv_backend=c10d --rdzv_endpoint=MASTER_ADDR:12355  genai/distrib.py
    """
    logger.info("Args: %s, Kwargs: %s", args, kwargs)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Start running basic DDP example on rank {rank}/{world_size}.")
    logger.info(f"local rank: {local_rank}")

    a = torch.ones(10).to(f"cuda:{rank}") * rank
    logger.info(f"Device of a in rank {rank}: {a.device}")
    avg = a.mean()
    logger.info(f"Rank {rank}: {a} | Mean: {avg}")

    dist.barrier()
    avg = dist.all_reduce(a)
    avg = a.mean() / world_size
    logger.info(f"Rank {rank}: {a} | Mean: {avg}")

    dist.destroy_process_group()
