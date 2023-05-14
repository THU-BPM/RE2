import json
def get_dataset(type="train", ds="semeval"):
    texts = json.load(open(f"./{ds}/{type}_sentence.json", "r"))
    relations = json.load(open(f"./{ds}/{type}_label_id.json", "r"))
    print(f"processing {type} dataset ...")
    return texts, relations