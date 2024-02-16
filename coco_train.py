
import numpy as np
import json
import os



def generate_metadata():
    result = {}
    with open('datasets/annotations/captions_train2017.json') as f:
        data = json.load(f)
        
        for i in data['images']:
            result[i['id']] = {}
            result[i['id']]['file_name'] = i['file_name']
        for i in data['annotations']:
            result[i['image_id']]['text'] = i['caption']
    return result

result = generate_metadata()
#with open('metadata.json', 'w') as f:   
#    json.dump(result,f,ensure_ascii=False)
final_result = []
for i in result.keys():
    final_result.append(json.dumps(result[i]))

with open('metadata.json', 'w') as f:   
    f.writelines(line + '\n' for line in final_result)