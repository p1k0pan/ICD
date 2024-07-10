import json
import os
import numpy as np
import pandas as pd
import argparse

def eval_pope(ans_file, label_file, output_file):
    answers = json.load(open(ans_file, 'r'))
    labels = json.load(open(label_file, 'r'))
    label_list = [item['label'] for item in labels]
    answers_copy = answers.copy()

    for answer in answers:
        text = answer['answer']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        elif 'Yes' in words or 'yes' in words:
            answer['answer'] = 'yes'
        else:
            answer['answer'] = answer['answer']

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        elif answer['answer'] == 'yes':
            pred_list.append(1)
        else:
            pred_list.append(-1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    fp_list = {}
    fn_list = {}
    tp_list = {}
    tn_list = {}    
    other = {}

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label, item in zip(pred_list, label_list, labels):
        ans = answers_copy[item['question_id']-1]['answer']
        item['prediction'] = ans
        if pred == -1:
            other[item['question_id']] = item
            continue
        if pred == pos and label == pos:
            TP += 1
            tp_list[item['question_id']] = item
        elif pred == pos and label == neg:
            FP += 1
            fp_list[item['question_id']] = item
        elif pred == neg and label == neg:
            TN += 1
            tn_list[item['question_id']] = item
        elif pred == neg and label == pos:
            FN += 1
            fn_list[item['question_id']] = item

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    results={'acc':[round(acc*100,2)], 'precision':[round(precision*100, 2)], 'recall':[round(recall*100,2)], 
             'f1':[round(f1*100,2)], 'yes_ratio':[round(yes_ratio*100,2)], 
              'tp':[TP],'fp':[FP],'tn':[TN], 'fn':[FN], 'other': [3000-TP-TN-FP-FN]}
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    save_csv(results, output_file, ans_file)
    return fp_list, fn_list, tp_list, tn_list, other

def save_csv(results, output_file, path):
    try:
        pre = path.split('/')
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
    results['format'] = [format]
    results['question'] = [question]
    results['prompt'] = [path.split('/')[-2]]
    df = pd.DataFrame(results)


    if os.path.exists(output_file):
        df.to_csv(output_file, mode="a", header=False,index=False)
    else:
        df.to_csv(output_file, header= True, index=False)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_folder", type=str, default="../gen_scripts/data/pope")
    parser.add_argument("--ans_folder", type=str, default="../gen_scripts/pope_results/ib/icd")
    parser.add_argument("--format", type=str, default="normal", choices=["no_format", "ow_format", "yn_format"])
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    data_files = {"coco_adversarial": f"{args.label_folder}/coco_pope_adversarial.json",
    "coco_popular": f"{args.label_folder}/coco_pope_popular.json",
    "coco_random": f"{args.label_folder}/coco_pope_random.json",
    "aokvqa_adversarial":f"{args.label_folder}/aokvqa_pope_seem_adversarial.json",
    "aokvqa_popular": f"{args.label_folder}/aokvqa_pope_seem_popular.json",
    "aokvqa_random": f"{args.label_folder}/aokvqa_pope_seem_random.json",
    "gqa_random": f"{args.label_folder}/gqfoldera_pope_seem_random.json",
    "gqa_popular": f"{args.label_folder}/gqfoldera_pope_seem_popular.json",
    "gqa_adversarial": f"{args.label_folder}/gqfoldera_pope_seem_adversarial.json"}

    if args.use_cd:
        sub_folder = "icd"
    else:
        sub_folder = "baseline"
    ans_folder = {
                   f"{args.ans_folder}/gqa_random/{args.format}": data_files["gqa_random"],
                   f"{args.ans_folder}/gqa_adversarial/{args.format}": data_files["gqa_adversarial"],
                   f"{args.ans_folder}/gqa_popular/{args.format}": data_files["gqa_popular"],
                    f"{args.ans_folder}/coco_adversarial/{args.format}": data_files["coco_adversarial"],
                   f"{args.ans_folder}/coco_popular/{args.format}": data_files["coco_popular"],
                   f"{args.ans_folder}/coco_random/{args.format}":  data_files["coco_random"],
                   f"{args.ans_folder}/aokvqa_adversarial/{args.format}": data_files["aokvqa_adversarial"],
                   f"{args.ans_folder}/aokvqa_popular/{args.format}": data_files["aokvqa_popular"],
                   f"{args.ans_folder}/aokvqa_random/{args.format}": data_files["aokvqa_random"],
        }

    for save_path, label_file in ans_folder.items():
        for path in os.listdir(save_path):
            if os.path.isdir(os.path.join(save_path, path)):
                print("process ", path)
                eval_path = os.path.join(save_path, path)
                if os.path.exists(os.path.join(eval_path, "normal.json")):
                    eval_file = os.path.join(eval_path, "normal.json")
                    fp_list, fn_list, tp_list, tn_list, other = eval_pope(eval_file, label_file, 
                                                                          os.path.join(save_path,'results.csv'))
                    prefix=""
                    json.dump(fp_list, open(os.path.join(eval_path,prefix +"fp_list.json"), 'w'))
                    json.dump(fn_list, open(os.path.join(eval_path,prefix +"fn_list.json"), 'w'))
                    json.dump(tp_list, open(os.path.join(eval_path,prefix +"tp_list.json"), 'w'))
                    json.dump(tn_list, open(os.path.join(eval_path,prefix +"tn_list.json"), 'w'))
                    if len(other) > 0:
                        json.dump(other, open(os.path.join(eval_path,prefix + "other.json"), 'w'))

                if os.path.exists(os.path.join(eval_path, "question.json")):
                    eval_file = os.path.join(eval_path, "question.json")
                    prefix="question_"
                    fp_list, fn_list, tp_list, tn_list, other = eval_pope(eval_file, label_file,
                                                                          os.path.join(save_path,'results.csv'))
                    json.dump(fp_list, open(os.path.join(eval_path,prefix +"fp_list.json"), 'w'))
                    json.dump(fn_list, open(os.path.join(eval_path,prefix +"fn_list.json"), 'w'))
                    json.dump(tp_list, open(os.path.join(eval_path,prefix +"tp_list.json"), 'w'))
                    json.dump(tn_list, open(os.path.join(eval_path,prefix +"tn_list.json"), 'w'))
                    if len(other) > 0:
                        json.dump(other, open(os.path.join(eval_path,prefix + "other.json"), 'w'))
