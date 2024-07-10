
import json
import spacy
nlp = spacy.load("en_core_web_sm")
import os
import pandas as pd
import argparse

def eval_objects(preds_path, gt, obj, output_file, prompt):
    # TP, TN, FP, FN = 0, 0, 0, 0
    preds = json.loads(open(preds_path, 'r').read())
    gt_obj = 0
    yes_ratio = 0
    result = {}
    other = {}
    result[obj] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "gt_obj": 0, "yes_ratio": 0}
    other[obj] = []
        
    
    for d in gt:
        for answer in preds:
            if d.get("image_id") == answer["image"]:
                text = answer["answer"]

                # Only keep the first sentence
                if text.find('.') != -1:
                    text = text.split('.')[0]

                text = text.replace(',', '')
                words = text.split(' ')
                if 'No' in words or 'not' in words or 'no' in words:
                    answer["answer"] = 'no'
                elif 'Yes' in words or 'yes' in words:
                    answer["answer"] = 'yes'
                    result[answer['object']]['yes_ratio'] += 1
                else:
                    answer["answer"] = answer["answer"]
                    other[answer['object']].append(answer)

                gt_objects = d.get("objects")
                if answer['object'] in gt_objects: # co_obj in this image
                    result[answer['object']]['num_co_obj'] += 1
                    if answer["answer"] =='yes': # answer yes
                        result[answer['object']]["TP"]+=1
                    elif answer["answer"] == "no": # answer no
                        result[answer['object']]["FN"]+=1
                else: # obj not in this image
                    if answer["answer"] =='yes':
                        result[answer['object']]["FP"]+=1
                    elif answer["answer"] == "no":
                        result[answer['object']]["TN"]+=1
    for o in objs:
        TP = result[o]['TP']
        FP = result[o]['FP']
        FN = result[o]['FN']
        TN = result[o]['TN']
        num_co_obj = result[o]['num_co_obj']
        yes_ratio = result[o]['yes_ratio']
        precision = float(result[o]['TP']) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2*precision*recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        yes_ratio = yes_ratio / len(preds) / len(objs)

        other_counts = len(other[o])
        save_results={'acc':[round(acc*100,2)], 'precision':[round(precision*100, 2)], 'recall':[round(recall*100,2)], 
                'f1':[round(f1*100,2)], 'yes_ratio':[round(yes_ratio*100,2)], 
                'tp':[TP],'fp':[FP],'tn':[TN], 'fn':[FN], 'other': [other_counts],
                'num_co_obj': num_co_obj, "num_gt_obj": len(gt)}
        
        try:
            pre = preds_path.split('/')
            if "yn_format" in pre:
                format = "yn_format"
            elif "ow_format" in pre:
                format = "ow_format"
            elif "no_format" in pre:
                format = "no_format"
            if "normal" in pre[-1]:
                question = "No"
            elif "question" in pre[-1]:
                question = "Yes"
        except:
            format = "No"
            question = "No"
        save_results['format'] = [format]
        save_results['question'] = [question]
        save_results["hal_ratio"] = [FP / (len(gt)-gt_obj)]
        save_results["co_object"] = [o]
        save_results["prompt"] = [prompt]
        print(len(preds))
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1 score: {}'.format(f1))
        print('Yes ratio: {}'.format(yes_ratio))

        df = pd.DataFrame(save_results)


        if os.path.exists(output_file):
            df.to_csv(output_file, mode="a", header=False,index=False)
        else:
            df.to_csv(output_file, header= True, index=False)
    return other

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_objects", type=str, default="../data/co_occur/gt_objects.json")
    parser.add_argument("--ans_folder", type=str, default="../gen_scripts/co_occur/ib/icd")
    parser.add_argument("--format", type=str, default="no_format", choices=["no_format", "ow_format", "yn_format"])
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    gt = json.loads(open(args.gt_objects, 'r').read())
    gt_obj = args.gt_objects.split(".")[-2].split("/")[-1].replace("gt_objects_", "")
    if gt_obj == "gt_objects":
        gt_name = "_all"
    else:
        gt_name = f"_{gt_obj}"

    objs = ["fork"]
    for obj in objs:
        path = f"{args.ans_folder}/co{gt_name}_{obj}/{args.format}"

        for file in os.listdir(path):

            if os.path.exists(os.path.join(path,file, "normal.json")):
                preds_path = os.path.join(path,file, "normal.json")
                other = eval_objects(preds_path, gt,obj, os.path.join(path, f'co{gt_name}_{obj}_eval.csv'), file)
                json.dump(other, open(os.path.join(path, file, "other_normal.json"), 'w'))
                
            if os.path.exists(os.path.join(path,file, "question.json")):
                preds_path = os.path.join(path,file, "question.json")
                other = eval_objects(preds_path, gt,obj, os.path.join(path, f'co{gt_name}_{obj}_eval.csv'), file)
                json.dump(other, open(os.path.join(path, file, "other_question.json"), 'w'))
