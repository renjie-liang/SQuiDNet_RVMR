import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.config import SharedOpt
from model.squidnet import SQuiDNet
from loader import SQDataset, SQTrainDataset
from inference import eval_epoch
from optim.adamw import AdamW
from utils.basic_utils import AverageMeter,load_config, get_logger
from utils.model_utils import count_parameters, set_cuda, vcmr_collate
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
import pdb



def rm_key_from_odict(odict_obj, rm_suffix):
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


def build_optimizer(model, opts):
    param_optimizer = [(n, p) for n, p in model.named_parameters() if (n.startswith('encoder') or n.startswith('query_weight')) and p.requires_grad ]
    param_top = [(n, p) for n, p in model.named_parameters() if  ( not n.startswith('encoder') and not n.startswith('query_weight'))  and p.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_top if not any(nd in n for nd in no_decay)], 'weight_decay': opts.wd},
        {'params': [p for n, p in param_top if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'lr': opts.lr_mul * opts.lr, 'weight_decay': opts.wd},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'lr': opts.lr_mul * opts.lr, 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opts.lr)
    return optimizer


def train(model, train_dataset, val_dataset, opt, logger):
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        #assert len(opt.device_ids) == 1
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU

    train_loader = DataLoader(train_dataset, collate_fn=vcmr_collate, batch_size=opt.batch, num_workers=opt.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    # train_eval_loader = DataLoader(train_eval_dataset, collate_fn=vcmr_collate, batch_size=opt.batch, num_workers=opt.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    # Prepare optimizer
    optimizer = build_optimizer(model, opt)
    # optimizer = None

    prev_best_score = 0.
    es_cnt = 0
    start_epoch = 0 if opt.no_eval_untrained else -1
    eval_step = len(train_loader) // opt.eval_num_per_epoch
    eval_tasks = opt.eval_tasks 
    save_submission_filename = "latest_{}_{}_predictions_{}.json".format(opt.data_name, opt.eval_type, "_".join(eval_tasks))

    for epoch in range(start_epoch, opt.n_epoch):
        model.train()
        num_training = len(train_loader)
        loss_meter = AverageMeter()
        for step, batch in tqdm(enumerate(train_loader), desc=f"Training", total=num_training):
            global_step = epoch * num_training + step + 1
            # continue
            model_inputs = set_cuda(batch["model_inputs"], opt.device)
            loss, loss_dict = model(model_inputs)
            optimizer.zero_grad()
            loss.backward()
            loss_meter.update(loss.item())
            if opt.grad_clip != -1:
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                if total_norm > opt.grad_clip:
                    logger.info("clipping gradient: {} with coefficient as {}".format(total_norm, opt.grad_clip / total_norm))
                optimizer.step()
            if step % opt.log_step == 0:
                logger.info(f"EPOCH {epoch}/{opt.n_epoch} | STEP: {step}|{len(train_loader)} | Loss: {loss_meter.avg:.4f}")
                loss_meter.reset()
                for i in range(torch.cuda.device_count()):
                    logger.info(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                    logger.info(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")


            if global_step % eval_step == 0 or step == len(train_loader):
                with torch.no_grad():
                    metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch(model, val_dataset, opt, save_submission_filename, tasks=eval_tasks, max_after_nms=100)
                logger.info("metrics_no_nms {}".format(pprint.pformat(rm_key_from_odict(metrics_no_nms, rm_suffix="by_type"), indent=4)))
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms, indent=4)))



def train_squid():
    opt = SharedOpt().parse()
    logger = get_logger(opt.results_path, opt.exp_id)

    logger.info("setup opt configuration...")
    # Fix seed
    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensuer the cuda is available
    torch.cuda.manual_seed_all(seed)

    opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str}: epch {epoch:03d} loss {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str}: epch {epoch:03d} metrics {eval_metrics_str}\n"
    
    data_config = load_config(opt.data_config)
    train_dataset = SQTrainDataset(config=data_config, neg_bmr_pred_num=opt.neg_bmr_pred_num, bmr_allowance=opt.bmr_allowance)
    eval_dataset = SQDataset(data_type=opt.eval_type, config=data_config, max_vid_len=opt.max_vid_len, max_query_len=opt.max_query_len, is_val=True, max_vcmr_video=opt.max_vcmr_video)

    model_config = load_config(opt.model_config)
    model = SQuiDNet(model_config, vid_dim=opt.vid_dim, text_dim=opt.text_dim, hidden_dim=opt.hidden_dim, lw_vid=opt.lw_vid, lw_st_ed=opt.lw_st_ed, loss_measure=opt.loss_measure)

    n_all, n_trainable = count_parameters(model)
    logger.info("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))

    logger.info("Start Training...")
    # train(model, train_dataset, train_eval_dataset, eval_dataset, opt)
    train(model, train_dataset, eval_dataset, opt, logger)
    return opt.results_dir, opt.eval_type


if __name__ == '__main__':
    model_dir, eval_type  = train_squid()

