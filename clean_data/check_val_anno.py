import h5py
from utils.basic_utils import load_jsonl, save_json
from tqdm import tqdm

# Load the validation data
val_data_path = 'data/data/text_data_ref/tvr_val_release_noun_predicate_back.jsonl'
val_data = load_jsonl(val_data_path)

# Extract query IDs
query_id_list = [i['desc_id'] for i in val_data]

# Open the HDF5 file
query_path = '/red/bianjiang/liang.renjie/RVMR/TVR-Ranking/data/features/query_bert.h5'
with h5py.File(query_path, "r") as desc_bert_h5:
    
    none_list = []
    
    # Iterate through query IDs and check existence in HDF5
    for desc_id in tqdm(query_id_list):
        if str(desc_id) not in desc_bert_h5:
            none_list.append(desc_id)

# Output the none_list or process it further as needed

new_anno = [ i for i in val_data if i['desc_id'] not in none_list]
save_json(new_anno, 'data/data/text_data_ref/tvr_val_release_noun_predicate.json')

print(len(query_id_list))
print(len(new_anno))

# print(f"IDs not found in HDF5 file: {none_list}")
