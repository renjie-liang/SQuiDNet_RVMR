import lmdb
import msgpack
from utils.basic_utils import load_json, save_json
from tqdm import tqdm

# Load query data
train_data_path = "./data/TVR-Ranking/train_top01_noun_verb.json"
# train_data_path = "./data/TVR-Ranking/test_noun_verb.json"
# train_data_path = "./data/TVR-Ranking/val_noun_verb.json"
train_bmr_path = "./data/TVR-Ranking/new_bmr_pred_lmdb"


query_data = load_json(train_data_path)
query_id_list = list(i['query_id'] for i in query_data)

# Open the LMDB database
bmr_env = lmdb.open(train_bmr_path, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
bmr_pred = bmr_env.begin(buffers=True)

# Iterate through query IDs
none_list = []
for query_id in tqdm(query_id_list):
    dump = bmr_pred.get(str(query_id).encode())
    
    # Check if the dump is None (meaning the key was not found)
    if dump is None:
        print(f"No predictions found for query ID: {query_id}")
        none_list.append(query_id)
        continue

    # Deserialize the predictions using msgpack if dump is not None
    bmr_predictions = msgpack.loads(dump)

    # Proceed with your logic for non-empty predictions

# Close the LMDB environment
bmr_env.close()
print(len(query_data))
new_anno = [ i for i in query_data if i['query_id'] not in none_list]
print(len(new_anno))
save_json(new_anno, 'data/TVR-Ranking/train_top01_noun_verb_cleaned.json')
# save_json(new_anno, 'data/TVR-Ranking/test_noun_verb_cleaned.json')
# save_json(new_anno, 'data/TVR-Ranking/val_noun_verb_cleaned.json')

### Train: 69317 -> 69063
### Test: 2781 -> 2768
### Val: 500 -> 497