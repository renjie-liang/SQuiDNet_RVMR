import lmdb
import msgpack
import os
from utils.basic_utils import load_json
from tqdm import tqdm

# Load the video duration index
video_duration_idx_path = './data/data/text_data_ref/tvr_video2dur_idx.json'
vid_data = load_json(video_duration_idx_path)

# Combine all video data into a single dictionary
all_vid_data = {}
for k, v in vid_data.items():
    all_vid_data.update(v)

# Create a mapping from index to video names
all_idx2vid = {v[1]: k for k, v in all_vid_data.items()}

# Paths to the original LMDB databases
lmdb_paths = [
    './data/data/bmr/bmr_prd_train_tvr',
    './data/data/bmr/bmr_prd_val_tvr',
    './data/data/bmr/bmr_prd_test_public_tvr'
]

# Open the new LMDB database to store processed data
output_lmdb_path = './data/TVR-Ranking/new_bmr_pred_lmdb'
new_bmr_env = lmdb.open(output_lmdb_path, map_size=int(1e12))  # Adjust map_size as necessary

# Combine processing and saving within a single loop
with new_bmr_env.begin(write=True) as txn:
    for lmdb_path in lmdb_paths:
        print(f"Processing {lmdb_path}")
        bmr_env = lmdb.open(lmdb_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        bmr_pred = bmr_env.begin(buffers=True)
        
        for k, v in tqdm(bmr_pred.cursor()):
            id_ = k.tobytes().decode()

            # Deserialize the predictions using msgpack
            bmr_predictions = msgpack.loads(v)

            new_bmr = []
            for p in bmr_predictions:
                vid, score = p
                vname = all_idx2vid.get(vid)
                if vname:
                    new_bmr.append([vname, score])

            # Serialize the new predictions and write to the new LMDB
            dump = msgpack.dumps(new_bmr)
            txn.put(id_.encode(), dump)
        
        bmr_env.close() # Close each LMDB environment after processing
print("write to new file")
# Example: Retrieve and print all data from the new LMDB database
with new_bmr_env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        id_ = key.decode()
        predictions = msgpack.loads(value)

# Close the new LMDB environment
new_bmr_env.close()
