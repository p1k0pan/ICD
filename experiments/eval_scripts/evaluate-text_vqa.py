import json
import os
import numpy as np
import pandas as pd
import spacy
from torch import le
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
import argparse

def calculate_cider(gts, res):
    cider = Cider()
    score, scores = cider.compute_score(gts, res)
    
    print(f"CIDEr Score: {score}")
    print("CIDEr scores: ", scores)
    return score, scores

def calculate_bleu(gts, res):
    bleu = Bleu()
    score, scores = bleu.compute_score(gts, res)
    
    print(f"bleu Score: {score}")
    return score, scores

nlp = spacy.load("en_core_web_sm")
def eval(ans_file, label_file, path, output_file):

    answers = json.load(open(ans_file, 'r'))
    labels = json.load(open(label_file, 'r'))["data"]

    labels_dict = {item["question_id"]: item["answers"] for item in labels}
    answers_dict = {item["question_id"]: [item["answer"]] for item in answers}

    #cider
    cider_score, cider_scores = calculate_cider(labels_dict, answers_dict)

    # bleu
    bleu_score, bleu_scores = calculate_bleu(labels_dict, answers_dict)

    cor = 0
    cor_file = []
    neg_file = []
    for answer in answers:
        text = answer['answer']
        answer["gt"]=[]

        doc = nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        lemmatized_text = ' '.join(lemmatized_tokens)

        question_id = answer["question_id"]
        label = labels_dict[question_id] # list of answers
        answer["gt"].append(label)

        if lemmatized_text in answer["gt"]:
            cor+=1
            cor_file.append(answer)
        else:
            flag = False
            for gt in answer["gt"]:
                if lemmatized_text in gt:
                    cor+=1
                    cor_file.append(answer)
                    flag=True
                    break
            if not flag:
                neg_file.append(answer)

    recall = cor/len(answers)
    print(recall)
    results= {"recall": recall, "cider": [cider_score], "b1": [bleu_score[0]], "b2": [bleu_score[1]], "b3": [bleu_score[2]], "b4": [bleu_score[3]], 
              "prompt": [path]}
    df = pd.DataFrame(results)


    if os.path.exists(output_file):
        df.to_csv(output_file, mode="a", header=False,index=False)
    else:
        df.to_csv(output_file, header= True, index=False)
    return cor_file, neg_file

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file", type=str, default="../data/text_vqa/TextVQA_0.5.1_val.json")
    parser.add_argument("--ans_folder", type=str, default="../gen_scripts/text_vqa/ib/icd")
    parser.add_argument("--format", type=str, default="normal", choices=["no_format", "ow_format", "yn_format"])
    return parser 

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    label_file = args.label_file
    save_path = f"{args.ans_folder}/{args.format}"

    for path in os.listdir(save_path):
        if os.path.isdir(os.path.join(save_path, path)):
            print("process ", path)
            eval_path = os.path.join(save_path, path)
            if os.path.exists(os.path.join(eval_path, "normal.json")):
                cor_file, neg_file = eval(os.path.join(eval_path, "normal.json"), 
                                                                    label_file, path, os.path.join(save_path,'metrics.csv'))
                prefix=""
                json.dump(cor_file, open(os.path.join(eval_path,prefix +"cor.json"), 'w'))
                json.dump(neg_file, open(os.path.join(eval_path,prefix +"neg.json"), 'w'))

            if os.path.exists(os.path.join(eval_path, "question.json")):
                prefix="question_"
                tn_list, other = eval(os.path.join(eval_path, "question.json"), 
                                                                    label_file, path, os.path.join(save_path,'metrics.csv'))
                json.dump(cor_file, open(os.path.join(eval_path,prefix +"cor.json"), 'w'))
                json.dump(neg_file, open(os.path.join(eval_path,prefix +"neg.json"), 'w'))