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
            with autocast(device_type='cuda'):
                model_inputs = set_cuda_half(batch_input["model_inputs"], device)
                outputs = model.MMAencoder.VSMMA(model_inputs)
            _video_feat = outputs['vid']
            _sub_feat = outputs['sub'] 
            
            all_video_feat.append(_video_feat.detach().cpu())
            all_video_mask.append(batch_input["model_inputs"]['vid']["feat_mask"].detach().cpu())
            all_sub_feat.append(_sub_feat.detach().cpu())
            # all_sub_mask.append(batch_input["model_inputs"]['sub']["feat_mask"].detach().cpu())

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
        with autocast(device_type='cuda'):
            model_inputs = set_cuda_half(batch_input["model_inputs"], device)
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

                part_start_probs = F.softmax(part_start_probs, dim=-1) 
                part_end_probs = F.softmax(part_end_probs, dim=-1)
                
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
        pre_start_time = [i[1].item() * args.max_vid_len for i in top_idx]
        pre_end_time   = [i[2].item() * args.max_vid_len for i in top_idx]
        
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