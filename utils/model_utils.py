__author__ = "Jie Lei"

#  ref: https://github.com/lichengunc/MAttNet/blob/master/lib/layers/lang_encoder.py#L11
#  ref: https://github.com/easonnie/flint/blob/master/torch_util.py#L272
import torch
from torch.utils.data.dataloader import  default_collate

N_Infinite = -1e10
P_Infinite = 1e10

def count_parameters(model):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_all, n_trainable


def set_cuda_local_rank(batch, local_rank):
    """
    Moves the batch to the appropriate CUDA device based on the local rank.
    """
    # local_rank = int(os.getenv('LOCAL_RANK', '0'))  # Get the local rank from the environment variable
    device = torch.device(f'cuda:{local_rank}')     # Create a device object for the specific local rank

    for key, value in batch.items():
        if isinstance(value, dict):
            for _key, _value in value.items():
                if isinstance(_value, torch.Tensor):
                    batch[key][_key] = _value.to(device, non_blocking=True)
        elif isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], torch.Tensor):
                    batch[key][i] = value[i].to(device, non_blocking=True)
        else:
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)

    return batch    

def set_cuda(batch,device):
    for key, value in batch.items():
        if isinstance(value, dict):
            for _key, _value in value.items():
                batch[key][_key] = _value.cuda(non_blocking=True, device=device)
        elif isinstance(value, (list,)):
            for i in range(len(value)):
                batch[key][i] = value[i].cuda(non_blocking=True, device=device)
        else:
            batch[key] = value.cuda(non_blocking=True, device=device)
    return batch


def set_cuda_half(batch,device):
    for key, value in batch.items():
        if isinstance(value, dict):
            for _key, _value in value.items():
                if isinstance(_value, torch.FloatTensor):
                    _value = _value.half()
                batch[key][_key] = _value.cuda(non_blocking=True, device=device)
        elif isinstance(value, (list,)):
            for i in range(len(value)):
                if isinstance(_value, torch.FloatTensor):
                    _value = _value.half()
                batch[key][i] = value[i].cuda(non_blocking=True, device=device)
        else:
            if isinstance(value, torch.FloatTensor):
                value = value.half()
            batch[key] = value.cuda(non_blocking=True, device=device)
    return batch


            


def mask_logits(target, mask):
    return target * mask + (1 - mask) * N_Infinite

# def vcmr_collate(batch):
#     batch_annotation = [e["annotation"] for e in batch]  # no need to collate
#     batch_data = default_collate([e["model_inputs"] for e in batch])
#     return {"annotation":batch_annotation, "model_inputs":batch_data}

def collate_fn(batch):
    batch_data = default_collate([e["model_inputs"] for e in batch])
    return {"model_inputs":batch_data}

