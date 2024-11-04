"""
SQuiD Loader
"""
import logging
import torch
from torch.utils.data import Dataset
import math
import os
import random
import lmdb
import io
import numpy as np
import json
from utils.basic_utils import load_jsonl, l2_normalize_np_array, load_json
import msgpack
import msgpack_numpy
import pdb
import sys
import h5py
logger = logging.getLogger(__name__)

def padding_feature(feature, max_feat_len):
    N, feat_dim = feature.shape
    feat_pad = torch.zeros((max_feat_len, feat_dim))
    feat_mask = torch.zeros(max_feat_len, dtype=torch.long)
    feat_pad[:N, :] = torch.from_numpy(feature)
    feat_mask[:N] = 1

    return feat_pad , feat_mask

def expand_annotations(annotations):
    new_annotations = []
    for i in annotations:
        query = i["query"]
        query_id = i["query_id"]
        noun = i["noun"]
        verb = i["verb"]
        for moment in  i["relevant_moment"]:
            moment.update({'query': query, 'query_id': query_id, 'noun': noun, 'verb': verb})
            new_annotations.append(moment)
    return new_annotations


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def get_query_feat(self, desc_id, token_id=1):
        query_feat = self.desc_bert_h5[str(desc_id)][:self.max_query_len]
        feat_pad, feat_mask = padding_feature(query_feat, self.max_query_len)

        tmp = dict()
        tmp["feat"] = feat_pad
        tmp["feat_mask"] = feat_mask
        tmp["feat_pos_id"] = torch.arange(self.max_query_len, dtype=torch.long)
        tmp["feat_token_id"] = torch.full((self.max_query_len,), token_id, dtype=torch.long)
        return tmp
    
    def get_vid_feat(self, vid_name):
        dump = self.vid_ft.get(vid_name.encode())
        img_dump = {k: np.copy(v) for k, v in msgpack_numpy.loads(dump, raw=False).items()}
        vid_feat = img_dump['features'][:self.max_vid_len]
        vid_feat = l2_normalize_np_array(vid_feat)
        return vid_feat
    
    def get_sub_feat(self, vid_name):
        dump = self.sub_bert_ft.get(vid_name.encode())
        with io.BytesIO(dump) as reader:
            feat_dump = np.load(reader, allow_pickle=True)
            sub_feat = feat_dump["features"][:self.max_vid_len]
        return sub_feat
    
    def get_nmr_bmr_vs_feat(self, vs_feat, nmr_bmr_vid_list, vs=True):
        L, feat_dim = vs_feat.shape
        nmr_bmr_feat_pad = torch.zeros((self.vidnum_per_q, self.max_vid_len, feat_dim))
        nmr_bmr_feat_mask = torch.zeros((self.vidnum_per_q, self.max_vid_len), dtype=torch.long)
        nmr_bmr_feat_pos_id = torch.repeat_interleave(torch.arange(self.max_vid_len, dtype=torch.long).unsqueeze(0), self.vidnum_per_q, dim=0)
        if vs:
            nmr_bmr_feat_token_id = torch.full((self.vidnum_per_q, self.max_vid_len), self.vid_token_id, dtype=torch.long) 
        else:
            nmr_bmr_feat_token_id = torch.full((self.vidnum_per_q, self.max_vid_len), self.text_token_id, dtype=torch.long) 

        for index, video_name in enumerate(nmr_bmr_vid_list,start=0):
            if vs:
                feat = self.get_vid_feat(video_name)
            else:
                feat = self.get_sub_feat(video_name)
            feat_pad, feat_mask = padding_feature(feat, self.max_vid_len)
            nmr_bmr_feat_pad[index] = feat_pad
            nmr_bmr_feat_mask[index] = feat_mask

        tmp = dict()
        tmp["feat"] = nmr_bmr_feat_pad
        tmp["feat_mask"] = nmr_bmr_feat_mask
        tmp["feat_pos_id"] = nmr_bmr_feat_pos_id
        tmp["feat_token_id"] = nmr_bmr_feat_token_id
        return tmp

    def get_bmr_pred(self, desc_id):
        dump = self.bmr_pred.get(str(desc_id).encode())
        bmr_predictions = msgpack.loads(dump)
        return bmr_predictions

    def SQDecision(self, nouns, verbs):
        n_num = len(nouns)
        v_num = len(verbs)
        topknouns = list(self.cctable.keys())[:10]
        is_positive = False
        for i in range(n_num):
            if nouns[i] in topknouns:
                for j in range(v_num):
                    topkverbs = self.cctable[nouns[i]]
                    #ForkedPdb().set_trace()
                    if verbs[j] == topkverbs[0][0]: # top 1
                        is_positive = True
        #is_positive = False
        return is_positive
    # Our recent further studie are implemented on Adaptive Range of negative samples for contrastive learning for the coocurrence table, which shows better performances.
    def SQuiDSample(self, bmr_preds, annotation, is_val):
        target_vidname = annotation["vid_name"]
        nouns = annotation["noun"]
        verbs = annotation["verb"]
        is_positive = self.SQDecision(nouns, verbs)
        loc = 100
        for idx, item in enumerate(bmr_preds):
            if target_vidname == item[0]:
                loc = idx
                break
        ##check all the location is below 100 when mode is train
        # assert  0<=loc<100

        vid_pool = [item[0] for item in bmr_preds if target_vidname != item[0]]
        if is_positive:
            sampled_vid = random.sample(vid_pool[:loc+int(self.bmr_allowance*0.1)], k=self.neg_bmr_pred_num)
        else:
            sampled_vid = random.sample(vid_pool[:loc+self.bmr_allowance], k=self.neg_bmr_pred_num)
        total_vid_name_list = [target_vidname,] + sampled_vid
        self.vidnum_per_q = 1 + self.neg_bmr_pred_num
        return total_vid_name_list, is_positive


class SQTrainDataset(BaseDataset):
    def __init__(self, data_path, config, max_vid_len=100, max_query_len=30, is_val=False, neg_bmr_pred_num=3, bmr_allowance=500, max_vcmr_video=10):

        raw_anno = load_json(data_path)
        self.query_data = expand_annotations(raw_anno)

        self.ground_truth = self.get_relevant_moment_gt(raw_anno)

        self.max_vid_len = max_vid_len
        self.max_query_len = max_query_len

        self.neg_bmr_pred_num = neg_bmr_pred_num
        self.is_val = is_val

        # query_data is for referencing
        self.desc_bert_h5 = h5py.File(config.query_path, "r")

        # ft is feature for query, sub and video
        self.bmr_env = lmdb.open(config.bmr_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.bmr_pred = self.bmr_env.begin(buffers=True)
        
        self.sub_bert_env = lmdb.open(config.sub_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.sub_bert_ft = self.sub_bert_env.begin(buffers=True)
        
        self.vid_env = lmdb.open(config.vid_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.vid_ft = self.vid_env.begin(buffers=True)
        
        with open(config.cctable_path, 'r') as ftmp:
            self.cctable = json.load(ftmp)

        self.data_root = config.data_path # ./data
        tmp_vid_data = load_json(os.path.join(self.data_root, config.vcmr_video_duration_idx_path))
        self.idx2vid = {}
        self.vid2idx = {}
        for _, set_data in tmp_vid_data.items():
            for vname, v in set_data.items():
                vidx = v[1]
                self.idx2vid[vidx] = vname
                self.vid2idx[vname] = vidx

        self.vname_duration = load_json(config.corpus_path)
        self.bmr_allowance = bmr_allowance
        self.max_vcmr_video = max_vcmr_video

        self.vid_token_id = 0
        self.text_token_id = 1
        self.vidnum_per_q = 1

    def __len__(self):
        return len(self.query_data)


    def __getitem__(self, index):

        ann = self.query_data[index]
        annotation = dict(desc_id=ann["query_id"], desc=ann["query"], vid_name=ann["video_name"], ts=ann["timestamp"], noun=ann["noun"], verb=ann["verb"])

        model_inputs = dict()
        ## get query feature (RoBerta 768 dim)
        model_inputs["query"] = self.get_query_feat(annotation["desc_id"], token_id=self.text_token_id)

        ## get BMR predictions per queries for negative or positive by SQuiD Decision
        bmr_preds = self.get_bmr_pred(annotation["desc_id"])
        is_positive = False
        total_vid_name_list, is_positive = self.SQuiDSample(bmr_preds, annotation, is_val=False)

        # sampled neg_bmr_pred_num negative videos or top-k videos
        annotation["max_vcmr_vid_name_list"] = total_vid_name_list[1:]

        model_inputs["is_positive"] = is_positive
        vid_feat = self.get_vid_feat(annotation["vid_name"])
        vid_L, feat_dim = vid_feat.shape
        model_inputs["vid"] = self.get_nmr_bmr_vs_feat(vid_feat, total_vid_name_list, vs=True)

        sub_feat = self.get_sub_feat(annotation["vid_name"])
        model_inputs["sub"] = self.get_nmr_bmr_vs_feat(sub_feat, total_vid_name_list, vs=False)

        max_vl = vid_L - 1
        start_idx = min(math.floor(annotation["ts"][0] / 1.5), max_vl)
        end_idx = min(math.ceil(annotation["ts"][1] / 1.5) - 1, max_vl)  # st_idx could be the same as ed_idx
        assert 0 <= start_idx <= end_idx <= max_vl, (annotation["ts"], start_idx, end_idx, max_vl)
        model_inputs["st_ed_indices"] =  torch.LongTensor([start_idx, end_idx])

        return dict(annotation=annotation, model_inputs=model_inputs)
    
    def get_relevant_moment_gt(self, annotations):
        gt_all = {}
        for data in annotations:
            gt_all[data["query_id"]] = data["relevant_moment"]
        return gt_all

class SQCorpusDataset(BaseDataset):
    def __init__(self, data_path, config, max_vid_len=100, bmr_allowance=500, max_vcmr_video=10):

        video_data = load_json(data_path)
        self.video_data = [{"vid_name": k, "duration": v} for k, v in video_data.items()]
        self.corpus_video_list = list(video_data.keys())

        self.max_vid_len = max_vid_len
        self.sub_bert_env = lmdb.open(config.sub_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.sub_bert_ft = self.sub_bert_env.begin(buffers=True)
        
        self.vid_env = lmdb.open(config.vid_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.vid_ft = self.vid_env.begin(buffers=True)

        self.bmr_allowance = bmr_allowance
        self.max_vcmr_video = max_vcmr_video

        self.vid_token_id = 0
        self.text_token_id = 1
        self.vidnum_per_q = 1

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, index):
        raw_data = self.video_data[index]
        # initialize with basic data
        duration = raw_data["duration"]
        video_name = raw_data["vid_name"]
        # meta = dict(vid_name=raw_data["vid_name"], duration=raw_data["duration"])
        model_inputs = dict()

        vid_feat = self.get_vid_feat(video_name)
        model_inputs["vid"] = self.get_nmr_bmr_vs_feat(vid_feat, [video_name], vs=True)
        sub_feat = self.get_sub_feat(video_name)
        model_inputs["sub"] = self.get_nmr_bmr_vs_feat(sub_feat, [video_name], vs=False)

        return dict(model_inputs=model_inputs)




class SQEvalDataset(BaseDataset):
    def __init__(self, data_path, config,  max_query_len=30):


        self.query_data = load_json(data_path)
        self.ground_truth = self.get_relevant_moment_gt(self.query_data)

        self.max_query_len = max_query_len

        self.desc_bert_h5 = h5py.File(config.query_path, "r")

        with open(config.cctable_path, 'r') as ftmp:
            self.cctable = json.load(ftmp)

        self.vname_duration = load_json(config.corpus_path)
        self.vid_token_id = 0
        self.text_token_id = 1
        self.vidnum_per_q = 1

    def __len__(self):
        return len(self.query_data)


    def __getitem__(self, index):

        ann = self.query_data[index]
        query_id = ann["query_id"]
        model_inputs = dict()
        model_inputs["query_id"] = query_id
        model_inputs["query"] = self.get_query_feat(query_id, token_id=self.text_token_id)

        
        return dict(model_inputs=model_inputs)

    def get_relevant_moment_gt(self, annotations):
        gt_all = {}
        for data in annotations:
            gt_all[data["query_id"]] = data["relevant_moment"]
        return gt_all



class SQDataset(Dataset):
    def __init__(self, config, max_vid_len=100, max_query_len=30, data_type="train", is_val=False, neg_bmr_pred_num=3, bmr_allowance=500, max_vcmr_video=10):
        
        # print(config)
        # print(config)
        # self.data_root = "./data/data" # ./data
        self.data_root = config.data_path # ./data
        self.query_ft_path = os.path.join(self.data_root, config.vcmr_query_path)
        self.sub_ft_path = config.sub_path
        self.vid_ft_path = config.vid_path
        self.type = data_type
        # In our further studies, we use bmr with hero, which shows better performances.
        if self.type == "train":
            self.data_path = os.path.join(self.data_root, config.train_data_path)
            self.bmr_pred_path = os.path.join(self.data_root, config.train_bmr_path)
        elif self.type == "val":
            self.data_path = os.path.join(self.data_root, config.vcmr_eval_data_path)
            self.bmr_pred_path = os.path.join(self.data_root, config.vcmr_eval_bmr_path)
        elif self.type == "test_public":
            self.data_path = os.path.join(self.data_root, config.test_data_path)
            self.bmr_pred_path = os.path.join(self.data_root, config.test_bmr_path)

        self.max_vid_len = max_vid_len
        self.max_query_len = max_query_len

        self.neg_bmr_pred_num = neg_bmr_pred_num
        self.is_val = is_val

        # query_data is for referencing
        self.query_data = load_jsonl(self.data_path)

        # ft is feature for query, sub and video
        self.bmr_env = lmdb.open(self.bmr_pred_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.bmr_pred = self.bmr_env.begin(buffers=True)
        
        self.query_env = lmdb.open(self.query_ft_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.query_ft = self.query_env.begin(buffers=True)
        
        self.sub_bert_env = lmdb.open(self.sub_ft_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.sub_bert_ft = self.sub_bert_env.begin(buffers=True)
        
        self.vid_env = lmdb.open(self.vid_ft_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.vid_ft = self.vid_env.begin(buffers=True)
        
        self.cctable_path = os.path.join(self.data_root, config.vcmr_cctable_path)
        with open(self.cctable_path, 'r') as ftmp:
            self.cctable = json.load(ftmp)

        vid_data = load_json(os.path.join(self.data_root, config.vcmr_video_duration_idx_path))[self.type]
        self.vid_data = [{"vid_name": k, "duration": v[0]} for k, v in vid_data.items()]
        self.vid2idx = {k: v[1] for k, v in vid_data.items()}
        self.idx2vid = {v[1]:k for k, v in vid_data.items()}
        self.bmr_allowance = bmr_allowance
        self.max_vcmr_video = max_vcmr_video

        self.vid_token_id = 0
        self.text_token_id = 1
        self.vidnum_per_q = 1

    def __len__(self):
        return len(self.query_data)



    def get_query_feat(self, desc_id, token_id=1):
        dump = self.query_ft.get(str(desc_id).encode())
        with io.BytesIO(dump) as reader:
            feat_dump = np.load(reader, allow_pickle=True)
            query_feat = feat_dump['features'][:self.max_query_len]
        feat_pad, feat_mask = padding_feature(query_feat, self.max_query_len)

        tmp = dict()
        tmp["feat"] = feat_pad
        tmp["feat_mask"] = feat_mask
        tmp["feat_pos_id"] = torch.arange(self.max_query_len, dtype=torch.long)
        tmp["feat_token_id"] = torch.full((self.max_query_len,), token_id, dtype=torch.long)
        return tmp
    
    def get_vid_feat(self, vid_name):
        dump = self.vid_ft.get(vid_name.encode())
        img_dump = {k: np.copy(v) for k, v in msgpack_numpy.loads(dump, raw=False).items()}
        vid_feat = img_dump['features'][:self.max_vid_len]
        vid_feat = l2_normalize_np_array(vid_feat)
        return vid_feat
    
    def get_sub_feat(self, vid_name):
        dump = self.sub_bert_ft.get(vid_name.encode())
        with io.BytesIO(dump) as reader:
            feat_dump = np.load(reader, allow_pickle=True)
            sub_feat = feat_dump["features"][:self.max_vid_len]
        return sub_feat
    
    def get_nmr_bmr_vs_feat(self, vs_feat, nmr_bmr_vid_list, vs=True):
        L, feat_dim = vs_feat.shape
        nmr_bmr_feat_pad = torch.zeros((self.vidnum_per_q, self.max_vid_len, feat_dim))
        nmr_bmr_feat_mask = torch.zeros((self.vidnum_per_q, self.max_vid_len), dtype=torch.long)
        nmr_bmr_feat_pos_id = torch.repeat_interleave(torch.arange(self.max_vid_len, dtype=torch.long).unsqueeze(0), self.vidnum_per_q, dim=0)
        if vs:
            nmr_bmr_feat_token_id = torch.full((self.vidnum_per_q, self.max_vid_len), self.vid_token_id, dtype=torch.long) 
        else:
            nmr_bmr_feat_token_id = torch.full((self.vidnum_per_q, self.max_vid_len), self.text_token_id, dtype=torch.long) 

        for index, video_name in enumerate(nmr_bmr_vid_list,start=0):
            if vs:
                feat = self.get_vid_feat(video_name)
            else:
                feat = self.get_sub_feat(video_name)
            feat_pad, feat_mask = padding_feature(feat, self.max_vid_len)
            nmr_bmr_feat_pad[index] = feat_pad
            nmr_bmr_feat_mask[index] = feat_mask

        tmp = dict()
        tmp["feat"] = nmr_bmr_feat_pad
        tmp["feat_mask"] = nmr_bmr_feat_mask
        tmp["feat_pos_id"] = nmr_bmr_feat_pos_id
        tmp["feat_token_id"] = nmr_bmr_feat_token_id
        return tmp

    def get_bmr_pred(self, desc_id):
        dump = self.bmr_pred.get(str(desc_id).encode())
        bmr_predictions = msgpack.loads(dump)
        return bmr_predictions

    def SQDecision(self, nouns, verbs):
        n_num = len(nouns)
        v_num = len(verbs)
        topknouns = list(self.cctable.keys())[:10]
        is_positive = False
        for i in range(n_num):
            if nouns[i] in topknouns:
                for j in range(v_num):
                    topkverbs = self.cctable[nouns[i]]
                    #ForkedPdb().set_trace()
                    if verbs[j] == topkverbs[0][0]: # top 1
                        is_positive = True
        #is_positive = False
        return is_positive
    # Our recent further studie are implemented on Adaptive Range of negative samples for contrastive learning for the coocurrence table, which shows better performances.
    def SQuiDSample(self, bmr_preds, annotation, is_val):
        target_vidname = annotation["vid_name"]
        nouns = annotation["noun"]
        verbs = annotation["verb"]
        is_positive = self.SQDecision(nouns, verbs)
        loc = 100
        for idx, item in enumerate(bmr_preds):
            if target_vidname == self.idx2vid[item[0]]:
                loc = idx
                break
        ##check all the location is below 100 when mode is train
        if self.type =="train":
            assert  0<=loc<100
        if is_val:
            # vcmr is performed on predictions from bmr
            first_vr_video_pool_list = [ self.idx2vid[item[0]] for item in bmr_preds[:self.max_vcmr_video]]
            total_vid_name_list = [target_vidname,] + first_vr_video_pool_list
            self.vidnum_per_q = 1 + self.max_vcmr_video
        else:
            vid_pool = [self.idx2vid[item[0]] for item in bmr_preds if target_vidname != self.idx2vid[item[0]] ]
            if is_positive:
                sampled_vid = random.sample(vid_pool[:loc+int(self.bmr_allowance*0.1)], k=self.neg_bmr_pred_num)
            else:
                sampled_vid = random.sample(vid_pool[:loc+self.bmr_allowance], k=self.neg_bmr_pred_num)
            total_vid_name_list = [target_vidname,] + sampled_vid
            self.vidnum_per_q = 1 + self.neg_bmr_pred_num
        return total_vid_name_list, is_positive

    def __getitem__(self, index):

        ann = self.query_data[index]
        annotation = dict(desc_id=ann["desc_id"], desc=ann["desc"], vid_name=ann["vid_name"] if "vid_name" in ann else None, ts=ann["ts"] if "ts" in ann else None, noun=ann["noun"] if "noun" in ann else None, verb=ann["verb"] if "verb" in ann else None)
        # For the test_public(challenge), video annotation per query is not public
        # dummy with no use
        if self.type =="test_public":
            annotation["vid_name"] = "friends_s01e01_seg02_clip_00"

        model_inputs = dict()
        ## get query feature (RoBerta 768 dim)
        model_inputs["query"] = self.get_query_feat(annotation["desc_id"], token_id=self.text_token_id)

        ## get BMR predictions per queries for negative or positive by SQuiD Decision
        bmr_preds = self.get_bmr_pred(annotation["desc_id"])
        is_positive = False
        if self.is_val:
            total_vid_name_list, is_positive = self.SQuiDSample(bmr_preds, annotation, self.is_val)
        else:
            total_vid_name_list, is_positive = self.SQuiDSample(bmr_preds, annotation, self.is_val)

        # sampled neg_bmr_pred_num negative videos or top-k videos
        annotation["max_vcmr_vid_name_list"] = total_vid_name_list[1:]

        model_inputs["is_positive"] = is_positive
        vid_feat = self.get_vid_feat(annotation["vid_name"])
        vid_L, feat_dim = vid_feat.shape
        model_inputs["vid"] = self.get_nmr_bmr_vs_feat(vid_feat, total_vid_name_list, vs=True)
        sub_feat = self.get_sub_feat(annotation["vid_name"])
        model_inputs["sub"] = self.get_nmr_bmr_vs_feat(sub_feat, total_vid_name_list, vs=False)

        if not self.is_val:
            max_vl = vid_L - 1
            start_idx = min(math.floor(annotation["ts"][0] / 1.5), max_vl)
            end_idx = min(math.ceil(annotation["ts"][1] / 1.5) - 1, max_vl)  # st_idx could be the same as ed_idx
            assert 0 <= start_idx <= end_idx <= max_vl, (annotation["ts"], start_idx, end_idx, max_vl)
            model_inputs["st_ed_indices"] =  torch.LongTensor([start_idx, end_idx])

        return dict(annotation=annotation, model_inputs=model_inputs)
