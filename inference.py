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
from utils.run_utils import logger_ndcg_iou, save_model, resume_model
from torch.amp import autocast, GradScaler
from lightning_fabric.utilities.seed import seed_everything




def infer(model, corpus_set, val_set, test_set, args, logger):

    scaler = GradScaler()
    corpus_loader = DataLoader(corpus_set, collate_fn=collate_fn, batch_size=args.local_batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, collate_fn=collate_fn, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, collate_fn=collate_fn, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    corpus_video_list = corpus_set.corpus_video_list
    val_gt = val_set.ground_truth
    test_gt = test_set.ground_truth

    corpus_feature = grab_corpus_feature(model, corpus_loader, args.device)
    val_ndcg_iou = eval_epoch(model, corpus_feature, val_loader, val_gt, args, corpus_video_list)
    # test_ndcg_iou = eval_epoch(model, corpus_feature, test_loader, test_gt, args, corpus_video_list)
    logger_ndcg_iou(val_ndcg_iou, logger, "VAL")
    # logger_ndcg_iou(test_ndcg_iou, logger, "TEST")


if __name__ == '__main__':
    args = SharedOpt().parse()
    seed_everything(args.seed)

    logger = get_logger(args.results_dir, args.exp)
    logger.info("setup args configuration...")
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)

    corpus_set = SQCorpusDataset(data_path=data_config.corpus_path, config=data_config)
    val_set = SQEvalDataset(data_path=data_config.val_data_path, config=data_config)
    test_set = SQEvalDataset(data_path=data_config.test_data_path, config=data_config)

    model = SQuiDNet(model_config, vid_dim=args.vid_dim, text_dim=args.text_dim, hidden_dim=args.hidden_dim, lw_vid=args.lw_vid, lw_st_ed=args.lw_st_ed, loss_measure=args.loss_measure)


    # Prepare optimizer
    if args.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(args.device)
        #assert len(args.device_ids) == 1
        # if len(args.device_ids) > 1:
        #     logger.info("Use multi GPU", args.device_ids)
        #     model = torch.nn.DataParallel(model, device_ids=args.device_ids)  # use multi GPU
    model, _, _ = resume_model(logger, args.load_dir, args.device, model=model)

    logger.info("Start Training...")
    infer(model, corpus_set, val_set, test_set, args, logger)



