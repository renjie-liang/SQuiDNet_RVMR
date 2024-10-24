import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import deepspeed
from config.config import SharedOpt
from model.squidnet import SQuiDNet
from loader import SQDataset, SQTrainDataset
from inference import eval_epoch
from optim.adamw import AdamW
from utils.basic_utils import AverageMeter,load_config, get_logger, rm_key_from_odict
from utils.model_utils import count_parameters, set_cuda, vcmr_collate, set_cuda_local_rank



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


def train(model, train_dataset, val_dataset, args, logger):
    # Prepare optimizer
    # if args.device.type == "cuda":
    #     logger.info("CUDA enabled.")
    #     model.to(args.device)
    #     #assert len(args.device_ids) == 1
    #     if len(args.device_ids) > 1:
    #         logger.info("Use multi GPU", args.device_ids)
    #         model = torch.nn.DataParallel(model, device_ids=args.device_ids)  # use multi GPU



    train_loader = DataLoader(train_dataset, collate_fn=vcmr_collate, batch_size=args.batch, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    # train_eval_loader = DataLoader(train_eval_dataset, collate_fn=vcmr_collate, batch_size=args.batch, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    # Prepare optimizer
    # optimizer = build_optimizer(model, args)
    # optimizer = None


    deepspeed.init_distributed()
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)

    model = model.to(local_rank)

    model_engine, optimizer, _, _ = deepspeed.initialize(config=args.deepspeed_config,
                                                         model=model,
                                                         model_parameters=model.parameters())

    prev_best_score = 0.
    es_cnt = 0
    start_epoch = 0 if args.no_eval_untrained else -1
    eval_interval = len(train_loader) // args.eval_interval_float
    eval_tasks = args.eval_tasks 
    save_submission_filename = "latest_{}_{}_predictions_{}.json".format(args.data_name, args.eval_type, "_".join(eval_tasks))

    for epoch in range(start_epoch, args.n_epoch):
        model.train()
        num_training = len(train_loader)
        loss_meter = AverageMeter()
        
        for step, batch in tqdm(enumerate(train_loader), desc=f"Training", total=num_training):
            global_step = epoch * num_training + step + 1
            # continue
            model_inputs = set_cuda_local_rank(batch["model_inputs"], local_rank)
            loss = model_engine(model_inputs)
            model_engine.backward(loss)
            model_engine.step()

            loss_meter.update(loss.item())
            if step % args.log_interval == 0:
                logger.info(f"EPOCH {epoch}/{args.n_epoch} | STEP: {step}|{len(train_loader)} | Loss: {loss_meter.avg:.4f}")
                loss_meter.reset()
                for i in range(torch.cuda.device_count()):
                    print(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                    print(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")


            if global_step % eval_interval == 0 or step == len(train_loader):
                with torch.no_grad():
                    metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch(model, val_dataset, args, save_submission_filename, tasks=eval_tasks, max_after_nms=100)
                logger.info("metrics_no_nms {}".format(pprint.pformat(rm_key_from_odict(metrics_no_nms, rm_suffix="by_type"), indent=4)))
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms, indent=4)))

                # save checkpoint
                # if metric_now > metric_best:
                #     client_sd['step'] = step
                #     ckpt_id = loss.item()
                #     model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)


# #load checkpoint
# _, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
# step = client_sd['step']

# #advance data loader to ckpt step
# dataloader_to_step(data_loader, step + 1)


def train_squid():
    args = SharedOpt().parse()
    logger = get_logger(args.results_path, args.exp_id)

    logger.info("setup args configuration...")
    # Fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensuer the cuda is available
    torch.cuda.manual_seed_all(seed)

    
    data_config = load_config(args.data_config)
    train_dataset = SQTrainDataset(config=data_config, neg_bmr_pred_num=args.neg_bmr_pred_num, bmr_allowance=args.bmr_allowance)
    eval_dataset = SQDataset(data_type=args.eval_type, config=data_config, max_vid_len=args.max_vid_len, max_query_len=args.max_query_len, is_val=True, max_vcmr_video=args.max_vcmr_video)

    model_config = load_config(args.model_config)
    model = SQuiDNet(model_config, vid_dim=args.vid_dim, text_dim=args.text_dim, hidden_dim=args.hidden_dim, lw_vid=args.lw_vid, lw_st_ed=args.lw_st_ed, loss_measure=args.loss_measure)

    n_all, n_trainable = count_parameters(model)
    logger.info("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))

    logger.info("Start Training...")
    # train(model, train_dataset, train_eval_dataset, eval_dataset, args)
    train(model, train_dataset, eval_dataset, args, logger)
    return args.results_dir, args.eval_type


if __name__ == '__main__':
    model_dir, eval_type  = train_squid()

