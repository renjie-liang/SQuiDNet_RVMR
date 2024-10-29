import os
import time
import torch
import argparse
import sys
import pprint
import json
from utils.basic_utils import load_json, save_json, make_zipfile
import pdb

def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


class SharedOpt(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = None

    def parser_init(self):
        self.parser.add_argument("--data_name", type=str, default="tvr", choices=["tvr", "didemo"])
        self.parser.add_argument("--eval_type", type=str, default="val", help="should be used for loss calculation and prediction")
        self.parser.add_argument("--results_dir", type=str, default="results")
        self.parser.add_argument("--exp", type=str, default=None, help="experiment name")
        self.parser.add_argument("--seed", type=int, default=2024, help="random seed")
        self.parser.add_argument("--device", type=int, default=0, help="0 means gpu id 0")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="use for multi gpu")
        self.parser.add_argument("--num_workers", type=int, default=8, help="num subprocesses used to load the data, 1 can debug loading time")

        # configuration: training specification
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_warmup_proportion", type=float, default=0.01, help="proportion to perform warm up of linear learning rate""0.1 = 10% of training.")
        self.parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=4000, help="number of epochs")
        self.parser.add_argument("--max_es_cnt", type=int, default=3,help="number of epochs to early stop, -1: no use of early stop")
        self.parser.add_argument("--global_batch_size", type=int, default=None, help="global batch size")
        self.parser.add_argument("--local_batch_size", type=int, default=None, help="local batch size")
        self.parser.add_argument("--no_eval_untrained", action="store_true", help="Evaluate for debug")
        self.parser.add_argument("--grad_clip", type=float, default=-1, help="perform gradient clip, -1: disable")
        self.parser.add_argument("--eval_epoch_num", type=int, default=1, help="eval_epoch_num")

        # configuration: data
        self.parser.add_argument("--max_vid_len", type=int, default=100, help="max number of vid len 100")
        self.parser.add_argument("--max_query_len", type=int, default=30, help="max number of words in query")
        self.parser.add_argument("--data_config", type=str,help="data config")

        # configuration: model
        self.parser.add_argument("--vid_dim", type=int,default=4352,help="video feature dimension")
        self.parser.add_argument("--text_dim", type=int, default=768, help="text feature dimension")
        self.parser.add_argument("--hidden_dim", type=int, default=768, help="joint feature  dimension")
        self.parser.add_argument("--model_config", type=str, help="model config")

        ## configuration: training vcmr
        self.parser.add_argument("--lw_st_ed", type=float, default=0.01, help="weight for moment level loss")
        self.parser.add_argument("--lw_vid", type=float, default=0.005, help="weight for video level loss")
        self.parser.add_argument("--lr_mul", type=float, default=1, help="Learning rate multiplier for backbone")
        self.parser.add_argument("--bmr_allowance", type=int, default=500, help="candidate for contrastive learninig to be used as positive or negativ")
        self.parser.add_argument("--neg_bmr_pred_num", type=int, default=15, help="multiple predictions of bmr can be used in contrastive learning")

        ## configuration: evaluation vcmr
        self.parser.add_argument("--loss_measure",type=str, choices=["moment", "moment_video"], default="moment", help="types of losses")

        # configuration: post processing
        self.parser.add_argument("--min_pred_l", type=int, default=0, help="minimum prediction (minimum moments)")
        self.parser.add_argument("--max_pred_l", type=int, default=24, help="maximum prediction (maximum moments)")
        self.parser.add_argument("--max_before_nms", type=int, default=200)
        self.parser.add_argument("--max_vcmr_video", type=int, default=10, help="ranking in top-max_vcmr_video")
        self.parser.add_argument("--nms_thd", type=float, default=-1, help="optinally use non-maximum suppression")
        # can use config files
        self.parser.add_argument('--config', help='JSON config file')
        self.parser.add_argument('--deepspeed_config', help='deepspeed JSON config file')
        self.parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
                
        self.parser.add_argument("--eval_folds", type=float, default=1.0, help="eval times during each epoch")
        self.parser.add_argument("--log_interval", type=int, default=100)
        self.parser.add_argument("--save_interval", type=int, default=200)
        self.parser.add_argument("--load_dir", type=str, default=None)
        self.parser.add_argument("--ckpt_id", type=str, default=None)
    

        self.parser.add_argument("--iou_threshold", type=float, nargs='+', default=[0.3, 0.5, 0.7], help="List of IOU thresholds")
        self.parser.add_argument("--ndcg_topk", type=int, nargs='+', default=[10, 20, 40], help="List of NDCG top k values")

    def parse(self):
        self.parser_init()
        args = parse_with_config(self.parser)

        args.results_dir = os.path.join(args.results_dir,"_".join([args.exp, time.strftime("%Y%m%d_%H%M%S")]))
        os.makedirs(args.results_dir, exist_ok=True)

        # args.device = torch.device("cuda:%d" % args.device_ids[0] if args.device >= 0 else "cpu")
        args.device = torch.device("cuda" if args.device >= 0 else "cpu")
        self.args = args
        return args


class TestOpt(SharedOpt):
    """add additional options for evaluating"""
    def parser_init(self):
        SharedOpt.parser_init(self)
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--model_dir", type=str, help="dir contains the model file, will be converted to absolute path afterwards")
        self.parser.add_argument("--tasks", type=str, nargs="+", choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"], help="tasks to run.")

if __name__ == '__main__':
    print(__file__)
    print(os.path.realpath(__file__))
    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print(code_dir)
