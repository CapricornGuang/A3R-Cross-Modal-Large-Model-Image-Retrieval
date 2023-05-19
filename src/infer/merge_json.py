import json

with open('open_clip_person_infer.json', 'r') as f1:
    data1 = json.load(f1)

with open('open_clip_car_infer.json', 'r') as f2:
    data2 = json.load(f2)

result_list = data2["results"] + data1["results"]

with open('submission.json', 'w') as f3:
    f3.write(json.dumps({'results': result_list}, indent=4))