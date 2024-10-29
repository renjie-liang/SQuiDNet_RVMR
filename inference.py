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
# from torch.utils.tensorboard import SummaryWriter
from config.config import SharedOpt
from model.squidnet import SQuiDNet
from loader import SQTrainDataset, SQCorpusDataset, SQEvalDataset
# from inference import eval_epoch
from optim.adamw import AdamW
from utils.basic_utils import AverageMeter,load_config, get_logger, rm_key_from_odict
from utils.model_utils import count_parameters, set_cuda, collate_fn, set_cuda_half
from model.infer_lib import grab_corpus_feature, eval_epoch
from utils.run_utils import logger_ndcg_iou, save_model
from torch.amp import autocast, GradScaler
from lightning_fabric.utilities.seed import seed_everything


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


def train(model, train_set, corpus_set, val_set, test_set, args, logger):

    scaler = GradScaler()
    train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=args.batch, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    corpus_loader = DataLoader(corpus_set, collate_fn=collate_fn, batch_size=args.batch, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, collate_fn=collate_fn, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, collate_fn=collate_fn, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    corpus_video_list = corpus_set.corpus_video_list
    val_gt = val_set.ground_truth
    test_gt = test_set.ground_truth

    # Prepare optimizer
    optimizer = build_optimizer(model, args)
    model = model.half()
    best_val_ndcg = 0.0
    start_epoch = 0 
    eval_step = int(len(train_loader) // args.eval_folds)
    for epoch in range(start_epoch, args.n_epoch):
        model.train()
        num_training = len(train_loader)
        loss_meter = AverageMeter()
        for step, batch in tqdm(enumerate(train_loader), desc=f"Training", total=num_training):
            step = step + 1
            # continue
            # model_inputs = set_cuda(batch["model_inputs"], args.device)

            with autocast(device_type='cuda'):
                optimizer.zero_grad()
                model_inputs = set_cuda_half(batch["model_inputs"], args.device)
                loss = model(model_inputs)
                print(loss)
                scaler.scale(loss).backward()
                # loss.backward()
                loss_meter.update(loss.item())
                optimizer.step()

            bestmodel_path = os.path.join(args.results_dir, "best_model.pt")
            save_model(model, optimizer, epoch, bestmodel_path, logger)
        
            # if step % args.log_interval == 0:
            #     logger.info(f"EPOCH {epoch}/{args.n_epoch} | STEP: {step}|{len(train_loader)} | Loss: {loss_meter.avg:.4f}")
            #     loss_meter.reset()
            #     for i in range(torch.cuda.device_count()):
            #         logger.info(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            #         logger.info(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

            # if step % eval_step == 0  or step == len(train_loader):
            #     corpus_feature = grab_corpus_feature(model, corpus_loader, args.device)
            #     val_ndcg_iou = eval_epoch(model, corpus_feature, val_loader, val_gt, args, corpus_video_list)
            #     # test_ndcg_iou = eval_epoch(model, corpus_feature, test_loader, test_gt, args, corpus_video_list)
            #     logger_ndcg_iou(val_ndcg_iou, logger, "VAL")
            #     # logger_ndcg_iou(test_ndcg_iou, logger, "TEST")

            #     if val_ndcg_iou[20][0.5] > best_val_ndcg:
            #         best_val_ndcg = val_ndcg_iou[20][0.5]
            #         logger_ndcg_iou(val_ndcg_iou, logger, "BEST VAL")
            #         # logger_ndcg_iou(test_ndcg_iou, logger, "BEST TEST")
            #         bestmodel_path = os.path.join(args.results_dir, "best_model.pt")
            #         save_model(model, optimizer, epoch, bestmodel_path, logger)


def train_squid():
    args = SharedOpt().parse()
    seed_everything(args.seed)

    logger = get_logger(args.results_dir, args.exp)
    logger.info("setup args configuration...")
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)


    train_set = SQTrainDataset(data_path=data_config.train_data_path, config=data_config, neg_bmr_pred_num=args.neg_bmr_pred_num, bmr_allowance=args.bmr_allowance)
    corpus_set = SQCorpusDataset(data_path=data_config.corpus_path, config=data_config)
    val_set = SQEvalDataset(data_path=data_config.val_data_path, config=data_config)
    test_set = SQEvalDataset(data_path=data_config.test_data_path, config=data_config)

    model = SQuiDNet(model_config, vid_dim=args.vid_dim, text_dim=args.text_dim, hidden_dim=args.hidden_dim, lw_vid=args.lw_vid, lw_st_ed=args.lw_st_ed, loss_measure=args.loss_measure)
    # resume_model(logger, args., model=None, optimizer=None, start_epoch=None):


    # Prepare optimizer
    if args.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(args.device)
        #assert len(args.device_ids) == 1
        if len(args.device_ids) > 1:
            logger.info("Use multi GPU", args.device_ids)
            model = torch.nn.DataParallel(model, device_ids=args.device_ids)  # use multi GPU

    logger.info("Start Training...")
    train(model, train_set, corpus_set, val_set, test_set, args, logger)


if __name__ == '__main__':
    model_dir, eval_type  = train_squid()

