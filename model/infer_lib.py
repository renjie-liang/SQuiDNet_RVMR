from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import numpy as np
import os
from utils.run_utils import topk_3d, generate_min_max_length_mask, extract_topk_elements
from model.ndcg_iou import calculate_ndcg_iou
from utils.model_utils import set_cuda, set_cuda_half
from torch.amp import autocast
from utils.basic_utils import save_json

def grab_corpus_feature(model, corpus_loader, device):
    all_video_feat, all_video_mask = [], []
    all_sub_feat, all_sub_mask = [], []
    
    with torch.no_grad():
        for batch_input in tqdm(corpus_loader, desc="Compute Corpus Feature: ", total=len(corpus_loader)):
            # with autocast(device_type='cuda'):
            # model_inputs = set_cuda_half(batch_input["model_inputs"], device)
            model_inputs = set_cuda(batch_input["model_inputs"], device)
            outputs = model.MMAencoder.VSMMA(model_inputs)
            _video_feat = outputs['vid']
            _sub_feat = outputs['sub'] 
            
            all_video_feat.append(_video_feat.detach().cpu())
            all_video_mask.append(batch_input["model_inputs"]['vid']["feat_mask"].detach().cpu())
            all_sub_feat.append(_sub_feat.detach().cpu())
            # all_sub_mask.append(batch_input["model_inputs"]['sub']["feat_mask"].detach().cpu())
            # breakpoint()
    all_video_feat = torch.cat(all_video_feat, dim=0)
    all_video_mask = torch.cat(all_video_mask, dim=0)
    all_sub_feat = torch.cat(all_sub_feat, dim=0)
    # all_sub_mask = torch.cat(all_sub_mask, dim=0)

    return  { "all_video_feat": all_video_feat,
              "all_video_mask": all_video_mask,
              "all_sub_feat": all_sub_feat,
            #   "all_sub_mask": all_sub_mask
              }


def eval_epoch(model, corpus_feature, eval_loader, eval_gt, args, corpus_video_list):
    topn_video = 50
    device = args.device
    model.eval()
    all_query_id = []
    all_video_feat = corpus_feature["all_video_feat"]
    all_video_mask = corpus_feature["all_video_mask"]
    all_sub_feat = corpus_feature["all_sub_feat"]
    # all_sub_mask = corpus_feature["all_sub_mask"]
    all_query_score, all_end_prob, all_start_prob, all_top_video_name = [], [], [], []

    for batch_input in tqdm(eval_loader, desc="Compute Query Scores: ", total=len(eval_loader)):
        # with autocast(device_type='cuda'):
            # model_inputs = set_cuda_half(batch_input["model_inputs"], device)
        # breakpoint()
        model_inputs = set_cuda(batch_input["model_inputs"], device)
        # get query feature and subtitle-matched video feature
        query_feature = model.MMAencoder.query_enc(model_inputs)
        query_batch = query_feature.shape[0]
            
        part_size = 300
        video_prediction_score, start_probs, end_probs = [], [], []
        for i in range(0, len(all_video_feat), part_size):
            part_video_feat = all_video_feat[i : i+part_size, :, :].to(device)
            part_video_mask = all_video_mask[i : i+part_size, :].to(device)
            part_sub_feat = all_sub_feat[i : i+part_size, :, :].to(device)

            vid_len = part_video_feat.shape[1]
            _query_feature, query_mask, tot_nmr_bmr_num = model.query_repeat(model_inputs, query_feature, part_video_feat)
            final_feat, res_feat = model.VQMMA_Plus(part_video_feat, part_sub_feat, _query_feature, part_video_mask, query_mask)
            part_start_probs, part_end_probs = model.CMP(final_feat, res_feat, part_video_mask)

            part_start_probs = part_start_probs.view(query_batch, tot_nmr_bmr_num, vid_len)
            part_end_probs = part_end_probs.view(query_batch, tot_nmr_bmr_num, vid_len)

            part_video_score = model.CMP.video_prediction(final_feat)
            part_video_score = part_video_score.view(query_batch, tot_nmr_bmr_num)

            # part_start_probs = F.softmax(part_start_probs, dim=-1) 
            # part_end_probs = F.softmax(part_end_probs, dim=-1)
            part_start_probs = F.sigmoid(part_start_probs)
            part_end_probs = F.sigmoid(part_end_probs)
            part_video_score = F.sigmoid(part_video_score)
            
            start_probs.append(part_start_probs.detach().cpu())
            end_probs.append(part_end_probs.detach().cpu())
            video_prediction_score.append(part_video_score.detach().cpu())

        start_probs = torch.concat(start_probs, dim=1)
        end_probs = torch.concat(end_probs, dim=1)
        video_prediction_score = torch.concat(video_prediction_score, dim=1)

        query_scores, start_probs,  end_probs, video_name_top = extract_topk_elements(video_prediction_score, start_probs, end_probs, corpus_video_list, topn_video)
        all_query_id.append(batch_input["model_inputs"]["query_id"].detach().cpu())
        all_query_score.append(query_scores.detach().cpu())
        all_start_prob.append(start_probs.detach().cpu())
        all_end_prob.append(end_probs.detach().cpu())
        all_top_video_name.extend(video_name_top)

        if len(all_query_id) > 10:
            break
    all_query_id = torch.cat(all_query_id, dim=0)
    all_query_id = all_query_id.tolist()
    
    all_query_score = torch.cat(all_query_score, dim=0)
    all_start_prob = torch.cat(all_start_prob, dim=0)
    all_end_prob = torch.cat(all_end_prob, dim=0)
    average_ndcg = calculate_average_ndcg(all_query_id, all_start_prob, all_query_score, all_end_prob, all_top_video_name, eval_gt, args)
    return average_ndcg

def calculate_average_ndcg(all_query_id, all_start_prob, all_query_score, all_end_prob, all_top_video_name, eval_gt, args):
    topn_moment = max(args.ndcg_topk)
    # breakpoint()
    all_2D_map = torch.einsum("qvm,qv,qvn->qvmn", all_start_prob, all_query_score, all_end_prob)
    map_mask = generate_min_max_length_mask(all_2D_map.shape, min_l=args.min_pred_l, max_l=args.max_pred_l)
    all_2D_map = all_2D_map * map_mask
    all_pred = {}
    for idx in trange(len(all_2D_map), desc="Collect Predictions: "):
        query_id = all_query_id[idx]
        score_map = all_2D_map[idx]
        top_score, top_idx = topk_3d(score_map, topn_moment)
        top_video_name  = all_top_video_name[idx]
        pred_videos = [top_video_name[i[0]] for i in top_idx]
        
        pre_start_time = [i[1].item() * 1.5       for i in top_idx]
        pre_end_time   = [i[2].item() * 1.5 + 1.5 for i in top_idx]
        # pre_start_time = [i[1].item() * args.max_vid_len for i in top_idx]
        # pre_end_time   = [i[2].item() * args.max_vid_len for i in top_idx]
        
        pred_result = []
        for video_name, s, e, score, in zip(pred_videos, pre_start_time, pre_end_time, top_score):
            pred_result.append({
                "video_name": video_name,
                "timestamp": [s, e],
                "model_scores": score.item()
            })
        # print(pred_result)
        all_pred[query_id] = pred_result

    save_json(all_pred, os.path.join(args.results_dir, "pred_results.json"))
    average_ndcg = calculate_ndcg_iou(eval_gt, all_pred, args.iou_threshold, args.ndcg_topk)
    return average_ndcg




def generate_min_max_mask(array_shape, min_l, max_l):
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    extra_length_mask_array = np.ones(mask_shape, dtype=np.float16) 
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask

from torch.utils.data import DataLoader
from utils.model_utils import set_cuda, vcmr_collate

def eval_rvmr(model, eval_dataset, args, max_before_nms=200, max_vcmr_video=100, maxtopk = 40):
    query_batch_size = args.local_eval_batch_size
    model.eval()
    query_eval_loader = DataLoader(eval_dataset, collate_fn=vcmr_collate, batch_size=query_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    n_total_query = len(eval_dataset)

    flat_st_ed_scores_sorted_indices = np.empty((n_total_query, max_before_nms), dtype=int)
    flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms), dtype=np.float32)
    sorted_q2c_indices = np.empty((n_total_query, max_vcmr_video), dtype=int)
    sorted_q2c_scores = np.empty((n_total_query, max_vcmr_video), dtype=np.float32)

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


    rvmr_res = {}
    for i, (_flat_st_ed_scores_sorted_indices, _flat_st_ed_sorted_scores) in tqdm(enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),desc="[VCMR]", total=n_total_query):
        video_indices_local, pred_st_indices, pred_ed_indices = np.unravel_index(_flat_st_ed_scores_sorted_indices, shape=(max_vcmr_video, args.max_vid_len, args.max_vid_len))
        video_indices = sorted_q2c_indices[i, video_indices_local]

        pred_st_in_seconds = pred_st_indices.astype(np.float32) * 1.5
        pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * 1.5 + 1.5
        max_vcmr_vid_name_pool = ann_info[i]["max_vcmr_vid_name_list"]
        cur_vcmr_redictions = []
        for j, (v_score, v_name_idx) in enumerate(zip(_flat_st_ed_sorted_scores, video_indices)):  # videos
            video_idx = max_vcmr_vid_name_pool[v_name_idx]
            cur_vcmr_redictions.append(
                    {
                    "video_name": video_idx,
                    "timestamp": [float(pred_st_in_seconds[j]), float(pred_ed_in_seconds[j])],
                    "model_scores": float(v_score)
                }
            )
        query_id = ann_info[i]['desc_id']
        rvmr_res[query_id] = cur_vcmr_redictions[:maxtopk]

    return rvmr_res
