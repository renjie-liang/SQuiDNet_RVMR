import os
import pprint
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from config.config import SharedOpt
from model.squidnet import SQuiDNet
from loader import SQTrainDataset, SQCorpusDataset, SQEvalDataset, SQDataset
# from inference import eval_epoch
from optim.adamw import AdamW
from utils.basic_utils import AverageMeter,load_config, get_logger, rm_key_from_odict, save_json
from utils.model_utils import count_parameters, set_cuda, collate_fn, set_cuda_half, vcmr_collate
from model.infer_lib import grab_corpus_feature, eval_epoch, eval_rvmr
from utils.run_utils import logger_ndcg_iou, save_model, resume_model
from lightning_fabric.utilities.seed import seed_everything
from model.ndcg_iou import calculate_ndcg_iou


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



def train(model, train_set, val_set, args, logger):

    train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=args.local_batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    # val_loader = DataLoader(val_set, collate_fn=vcmr_collate, batch_size=args.local_eval_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    # Prepare optimizer
    optimizer = build_optimizer(model, args)
    best_val_ndcg = 0.0
    start_epoch = 0 
    eval_step = int(len(train_loader) // args.eval_folds)
    for epoch in range(start_epoch, args.n_epoch):
        model.train()
        num_training = len(train_loader)
        loss_meter = AverageMeter()
        for step, batch in tqdm(enumerate(train_loader), desc=f"Training", total=num_training):
            step = step #+ 1
            optimizer.zero_grad()
            model_inputs = set_cuda(batch["model_inputs"], args.device)
            loss = model(model_inputs)
            print(loss)
            loss.backward()
            loss_meter.update(loss.item())
            optimizer.step()

            if step % args.log_interval == 0:
                logger.info(f"EPOCH {epoch}/{args.n_epoch} | STEP: {step}|{len(train_loader)} | Loss: {loss_meter.avg:.4f}")
                loss_meter.reset()
                for i in range(torch.cuda.device_count()):
                    logger.info(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                    logger.info(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")


            if step % eval_step == 0  or step == len(train_loader):
                pred_data = eval_rvmr(model, val_set, args, max_before_nms=args.max_before_nms, max_vcmr_video=40)
                gt_data = val_set.ground_truth
                val_ndcg_iou = calculate_ndcg_iou(gt_data, pred_data, args.iou_threshold, args.ndcg_topk)

                save_json(pred_data, os.path.join(args.results_dir, "rvmr_predictions.json"))
                logger_ndcg_iou(val_ndcg_iou, logger, "EVAL")
                    
                if val_ndcg_iou[20][0.5] > best_val_ndcg:
                    best_val_ndcg = val_ndcg_iou[20][0.5]
                    logger_ndcg_iou(val_ndcg_iou, logger, "BEST VAL")
                    bestmodel_path = os.path.join(args.results_dir, "best_model.pt")
                    save_model(model, optimizer, epoch, bestmodel_path, logger)

        save_model_path = os.path.join(args.results_dir, f"{epoch}_model.pt")
        save_model(model, optimizer, epoch, save_model_path, logger)



if __name__ == '__main__':
    args = SharedOpt().parse()
    seed_everything(args.seed)

    logger = get_logger(args.results_dir, args.exp)
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)

    logger.info(f"Args configuration:\n{pprint.pformat(vars(args), indent=4)}\n")
    logger.info(f"Data configuration:\n{pprint.pformat(data_config, indent=4)}\n")

    train_set = SQTrainDataset(data_path=data_config.train_data_path, config=data_config, neg_bmr_pred_num=args.neg_bmr_pred_num, bmr_allowance=args.bmr_allowance)
    val_set = SQTrainDataset(data_path=data_config.val_data_path, config=data_config, neg_bmr_pred_num=40, bmr_allowance=args.bmr_allowance)
    model = SQuiDNet(model_config, vid_dim=args.vid_dim, text_dim=args.text_dim, hidden_dim=args.hidden_dim, lw_vid=args.lw_vid, lw_st_ed=args.lw_st_ed, loss_measure=args.loss_measure)

    n_all, n_trainable = count_parameters(model)
    logger.info("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))

    # Prepare optimizer
    if args.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(args.device)
        #assert len(args.device_ids) == 1
        # if len(args.device_ids) > 1:
        #     logger.info("Use multi GPU", args.device_ids)
        #     model = torch.nn.DataParallel(model, device_ids=args.device_ids)  # use multi GPU



    logger.info("Start Training...")
    train(model, train_set, val_set, args, logger)


