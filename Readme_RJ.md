conda install -f requirements.yml


```
# Inference with the Biased Moment Retrieval (BMR). 
# BMR results contain a candidate videos list for each query, provided by this work.
# Therefore, we predict the moment from the candidate list. In implementation, the list contains 10 videos.
train_infer_all_bmr.py
```


```
# Inference by all videos. We calculate the score for all videos for each query, as in a normal VCMR pipeline. 
# However, the time cost is substantial due to the heavy modality fusion calculations.
train_infer_all_videos.py
```

**Quick run**
```
sbatch top01_run.slurm
```
