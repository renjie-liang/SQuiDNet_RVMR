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
from loader import SQTrainDataset, SQCorpusDataset, SQEvalDataset, SQDataset
# from inference import eval_epoch
from optim.adamw import AdamW
from utils.basic_utils import AverageMeter,load_config, get_logger, rm_key_from_odict, save_json
from utils.model_utils import count_parameters, set_cuda, collate_fn, set_cuda_half, vcmr_collate
from model.infer_lib import grab_corpus_feature, eval_epoch
from utils.run_utils import logger_ndcg_iou, save_model, resume_model
from torch.amp import autocast, GradScaler
from lightning_fabric.utilities.seed import seed_everything
import torch.nn.functional as F
from standalone_eval.eval import eval_retrieval
from unuse.inference import svmr_st_ed_probs, generate_min_max_mask
from utils.inference_utils  import get_submission_top_n, post_processing_vcmr_nms



def compute_query2vid(model, eval_dataset, args, max_before_nms=200, max_vcmr_video=100, tasks=("VCMR", "VR")):
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks
    is_svmr = "SVMR" in tasks

    vid2idx = eval_dataset.vid2idx
    query_batch_size = 8

    model.eval()
    query_eval_loader = DataLoader(eval_dataset, collate_fn=vcmr_collate, batch_size=query_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    n_total_query = len(eval_dataset)

    if is_vcmr:
        flat_st_ed_scores_sorted_indices = np.empty((n_total_query, max_before_nms), dtype=int)
        flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms), dtype=np.float32)
    if is_vr :
        sorted_q2c_indices = np.empty((n_total_query, max_vcmr_video), dtype=int)
        sorted_q2c_scores = np.empty((n_total_query, max_vcmr_video), dtype=np.float32)
    if is_svmr:
        svmr_gt_st_probs = np.zeros((n_total_query, args.max_vid_len), dtype=np.float32)
        svmr_gt_ed_probs = np.zeros((n_total_query, args.max_vid_len), dtype=np.float32)

    ann_info = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

        ann_info.extend(batch["annotation"])
        model_inputs = set_cuda(batch["model_inputs"], args.device)
        video_similarity_score, begin_score_distribution, end_score_distribution = model.get_pred_from_raw_query(model_inputs)

        _vcmr_st_prob = begin_score_distribution[:, 1:]
        _vcmr_ed_prob = end_score_distribution[:, 1:]

        video_similarity_score = video_similarity_score[:, 1:] # first element holds ground-truth information
        _query_context_scores = torch.softmax(video_similarity_score,dim=1)
        _sorted_q2c_scores, _sorted_q2c_indices = torch.topk(_query_context_scores, max_vcmr_video, dim=1, largest=True)

        if is_vr:
            sorted_q2c_indices[idx*query_batch_size : (idx+1)*query_batch_size] = _sorted_q2c_indices.detach().cpu().numpy()
            sorted_q2c_scores[idx*query_batch_size : (idx+1)*query_batch_size] = _sorted_q2c_scores.detach().cpu().numpy()


        _st_probs = F.softmax(_vcmr_st_prob, dim=-1)  # (query_batch, video_corpus, vid_len)
        _ed_probs = F.softmax(_vcmr_ed_prob, dim=-1)

        row_indices = torch.arange(0, len(_st_probs), device=args.device).unsqueeze(1)
        _st_probs = _st_probs[row_indices, _sorted_q2c_indices] 
        _ed_probs = _ed_probs[row_indices, _sorted_q2c_indices]

        _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs, _sorted_q2c_scores, _ed_probs)

        valid_prob_mask = generate_min_max_mask(_st_ed_scores.shape, min_l=args.min_pred_l, max_l=args.max_pred_l)

        _st_ed_scores *= torch.from_numpy(valid_prob_mask).to(_st_ed_scores.device)

        _n_q  = _st_ed_scores.shape[0]

        _flat_st_ed_scores = _st_ed_scores.reshape(_n_q, -1)
        _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = torch.sort(_flat_st_ed_scores, dim=1, descending=True)

        flat_st_ed_sorted_scores[idx*query_batch_size : (idx+1)*query_batch_size] = _flat_st_ed_sorted_scores[:, :max_before_nms].detach().cpu().numpy()
        flat_st_ed_scores_sorted_indices[idx*query_batch_size : (idx+1)*query_batch_size] = _flat_st_ed_scores_sorted_indices[:, :max_before_nms].detach().cpu().numpy()

    vr_res = []
    if is_vr:
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(enumerate(zip(sorted_q2c_scores, sorted_q2c_indices)), desc="[VR]", total=n_total_query):
            vr_predictions = []
            max_vcmr_vid_name_pool = ann_info[i]["max_vcmr_vid_name_list"]
            for j, (v_score, v_name_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = vid2idx[max_vcmr_vid_name_pool[v_name_idx]]
                vr_predictions.append([video_idx, 0, 0, float(v_score)])
            cur_pred = dict(desc_id=ann_info[i]["desc_id"], desc=ann_info[i]["desc"], predictions=vr_predictions)
            vr_res.append(cur_pred)

    svmr_res = []
    if is_svmr:
        svmr_res = svmr_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, ann_info, vid2idx, min_pred_l=args.min_pred_l, max_pred_l=args.max_pred_l, max_before_nms=max_before_nms)


    vcmr_res = []
    if is_vcmr:
        for i, (_flat_st_ed_scores_sorted_indices, _flat_st_ed_sorted_scores) in tqdm(enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),desc="[VCMR]", total=n_total_query):
            video_indices_local, pred_st_indices, pred_ed_indices = np.unravel_index(_flat_st_ed_scores_sorted_indices, shape=(max_vcmr_video, args.max_vid_len, args.max_vid_len))
            video_indices = sorted_q2c_indices[i, video_indices_local]

            pred_st_in_seconds = pred_st_indices.astype(np.float32) * 1.5
            pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * 1.5 + 1.5
            vcmr_predictions = []
            max_vcmr_vid_name_pool = ann_info[i]["max_vcmr_vid_name_list"]
            for j, (v_score, v_name_idx) in enumerate(zip(_flat_st_ed_sorted_scores, video_indices)):  # videos
                video_idx = vid2idx[max_vcmr_vid_name_pool[v_name_idx]]
                vcmr_predictions.append([video_idx, float(pred_st_in_seconds[j]), float(pred_ed_in_seconds[j]), float(v_score)])
            cur_pred = dict(desc_id=ann_info[i]["desc_id"], desc=ann_info[i]["desc"], predictions=vcmr_predictions)
            vcmr_res.append(cur_pred)

    res = dict(VCMR=vcmr_res, SVMR=svmr_res, VR=vr_res)
    return {k: v for k, v in res.items() if len(v) != 0}




def infer(model, eval_dataset, args, logger):
    eval_res = compute_query2vid(model, eval_dataset, args, max_before_nms=args.max_before_nms, max_vcmr_video=args.max_vcmr_video)
    eval_res["vid2idx"] = eval_dataset.vid2idx

    IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(args.results_dir, "vcmr_predictions.json")
    eval_submission = get_submission_top_n(eval_res, top_n=100)
    save_json(eval_submission, submission_path)

    metrics = eval_retrieval(eval_submission, eval_dataset.query_data, iou_thds=IOU_THDS, match_number=True, verbose=False, use_desc_type=args.data_name == "tvr")
    print(metrics)

        
    # scaler = GradScaler()
    # corpus_loader = DataLoader(corpus_set, collate_fn=collate_fn, batch_size=args.local_batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    # val_loader = DataLoader(val_set, collate_fn=collate_fn, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    # test_loader = DataLoader(test_set, collate_fn=collate_fn, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    # corpus_video_list = corpus_set.corpus_video_list
    # val_gt = val_set.ground_truth
    # test_gt = test_set.ground_truth

    # corpus_feature = grab_corpus_feature(model, corpus_loader, args.device)
    # val_ndcg_iou = eval_epoch(model, corpus_feature, val_loader, val_gt, args, corpus_video_list)
    # # test_ndcg_iou = eval_epoch(model, corpus_feature, test_loader, test_gt, args, corpus_video_list)
    # logger_ndcg_iou(val_ndcg_iou, logger, "VAL")
    # # logger_ndcg_iou(test_ndcg_iou, logger, "TEST")


if __name__ == '__main__':
    args = SharedOpt().parse()
    seed_everything(args.seed)

    logger = get_logger(args.results_dir, args.exp)
    logger.info("setup args configuration...")
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)

    eval_dataset = SQDataset(data_type=args.eval_type, config=data_config, max_vid_len=args.max_vid_len, max_query_len=args.max_query_len, is_val=True, max_vcmr_video=args.max_vcmr_video)

    # corpus_set = SQCorpusDataset(data_path=data_config.corpus_path, config=data_config)
    # val_set = SQEvalDataset(data_path=data_config.val_data_path, config=data_config)
    # test_set = SQEvalDataset(data_path=data_config.test_data_path, config=data_config)
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
    infer(model, eval_dataset, args, logger)



