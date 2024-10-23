srun --partition=gpu --gres=gpu:4 --nodes=1 --cpus-per-task=8 --mem=100gb --time=01:00:00 --account=bianjiang --qos=bianjiang --reservation=bianjiang --pty bash -i
micromamba activate squid
sh scripts/train.sh 
sh scripts/train_tvrr.sh 


/red/bianjiang/liang.renjie/RVMR/TVR-Ranking/data/features



## Dataset

```
batch = {'annotation': 
            [
                {'desc_id': ,
                'desc': ,
                'vid_name': ,
                'ts': ,
                'noun': ,
                'verb': ,
                'max_vcmr_vid_name_list': ,
                }
            ], (list)
        'model_inputs': 
            {
                'query': 
                    {'feat':            torch.Size([16, 30, 768]),
                    'feat_mask':        torch.Size([16, 30]),
                     'feat_pos_id':     torch.Size([16, 30]),           # [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
                     'feat_token_id':   torch.Size([16, 30])}           # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                'is_positive':      torch.Size([16]),                   # [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
                'vid': 
                    {'feat':            torch.Size([16, 16, 100, 4352]),
                     'feat_mask', 
                     'feat_pos_id', 
                     'feat_token_id'},
                'sub':
                    {'feat',            torch.Size([16, 16, 100, 768])
                     'feat_mask', 
                     'feat_pos_id', 
                     'feat_token_id'}
                'st_ed_indices':    torch.Size([16, 2]),
            }
        }


```

query_feature = self.MMAencoder.query_enc(features=batch["query"]["feat"], 
                                            position_ids=batch["query"]["feat_pos_id"], 
                                            token_type_ids=batch["query"]["feat_token_id"], 
                                            attention_mask=batch["query"]["feat_mask"])

vsMMA_feature = self.MMAencoder.VSMMA(vid_features=batch["vid"]["feat"],
                                        vid_position_ids=batch["vid"]["feat_pos_id"],
                                        vid_token_type_ids=batch["vid"]["feat_token_id"],
                                        vid_attention_mask=batch["vid"]["feat_mask"],
                                        text_features=batch["sub"]["feat"],
                                        text_position_ids=batch["sub"]["feat_pos_id"],
                                        text_token_type_ids=batch["sub"]["feat_token_id"],
                                        text_attention_mask=batch["sub"]["feat_mask"]

