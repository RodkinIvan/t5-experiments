import functools
import importlib
import inspect
import json
import logging
import os
import platform
import subprocess
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

import horovod.torch as hvd
import torch
import transformers

import lm_experiments_tools.optimizers


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def get_git_hash_commit() -> str:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        # no git installed or we are not in repository
        commit = ''
    return commit


def get_git_diff() -> str:
    try:
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary']).decode('utf8')
    except subprocess.CalledProcessError:
        # no git installed or we are not in repository
        diff = ''
    return diff


def get_fn_param_names(fn) -> List[str]:
    """get function parameters names except *args, **kwargs

    Args:
        fn: function or method

    Returns:
        List[str]: list of function parameters names
    """
    params = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]:
            params += [p.name]
    return params


def get_optimizer(name: str):
    if ':' in name:
        return get_cls_by_name(name)
    if hasattr(lm_experiments_tools.optimizers, name):
        return getattr(lm_experiments_tools.optimizers, name)
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)
    if hasattr(transformers.optimization, name):
        return getattr(transformers.optimization, name)
    try:
        apex_opt = importlib.import_module('apex.optimizers')
        return getattr(apex_opt, name)
    except (ImportError, AttributeError):
        pass
    return None


def collect_run_configuration(args, env_vars=['CUDA_VISIBLE_DEVICES']):
    args_dict = dict(vars(args))
    args_dict['ENV'] = {}
    for env_var in env_vars:
        args_dict['ENV'][env_var] = os.environ.get(env_var, '')
    args_dict['HVD_INIT'] = hvd.is_initialized()
    if hvd.is_initialized():
        args_dict['HVD_SIZE'] = hvd.size()
    args_dict['MACHINE'] = platform.node()
    args_dict['COMMIT'] = get_git_hash_commit()
    return args_dict


def get_distributed_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    if hvd.is_initialized():
        return hvd.rank()
    return 0


def rank_0(fn):
    @functools.wraps(fn)
    def rank_0_wrapper(*args, **kwargs):
        if get_distributed_rank() == 0:
            return fn(*args, **kwargs)
        return None
    return rank_0_wrapper


def prepare_run(args, logger=None, logger_fmt: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                add_file_logging=True):
    """creates experiment directory, saves configuration and git diff, setups logging

    Args:
        args: arguments parsed by argparser, model_path is a required field in args
        logger: python logger object
        logger_fmt (str): string with logging format
        add_file_logging (bool): whether to write logs into files or not
    """

    # create model path and save configuration
    rank = get_distributed_rank()
    if rank == 0 and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path / 'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())

    # configure logging to a file
    if args.model_path is not None and logger is not None and add_file_logging:
        # todo: make it independent from horovod
        if hvd.is_initialized():
            # sync workers to make sure that model_path is already created by worker 0
            hvd.barrier()
        # RotatingFileHandler will keep logs only of a limited size to not overflow available disk space.
        # Each gpu worker has its own logfile.
        # todo: make logging customizable? reconsider file size limit?
        fh = RotatingFileHandler(Path(args.model_path) / f"{time.strftime('%Y.%m.%d_%H:%M:%S')}_rank_{rank}.log",
                                 mode='w', maxBytes=100*1024*1024, backupCount=2)
        fh.setLevel(logger.level)
        fh.setFormatter(logging.Formatter(logger_fmt))
        logger.addHandler(fh)

    if rank == 0 and args.model_path is None and logger is not None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')


import subprocess as sp
NVIDIA_SMI_COMMAND = 'nvidia-smi --query-gpu=memory.used --format=csv'
def get_gpu_memory_usage(gpu_ids=None):
    sp_out = sp.check_output(NVIDIA_SMI_COMMAND.split())
    gpu_mem = str(sp_out).split('\\n')[1:-1]
    gpu_mem = list(map(lambda x: int(x.split(' MiB')[0]), gpu_mem))

    if gpu_ids is None:
        gpu_ids = range(len(gpu_mem))
    gpu_names = [f'used_memory_gpu_{i}' for i in gpu_ids]
    out = dict(zip(gpu_names, gpu_mem))
    return out