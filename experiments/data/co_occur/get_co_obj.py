import json
gt = json.loads(open("gt_objects.json", 'r').read())
co_obj = "fork"
result=[]
for d in gt:
    if co_obj in d.get("objects"):
        result.append(d)
print(len(result))
json.dump(result, open(f"gt_objects_{co_obj}.json", 'w'))