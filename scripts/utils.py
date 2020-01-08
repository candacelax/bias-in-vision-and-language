import json

def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        captions = v["captions"]
        data[k] = captions
        v["captions"] = captions
    return all_data
