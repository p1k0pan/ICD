import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import sys
import json
from collections import defaultdict
import pandas as pd

class OutputLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 这个方法是为了兼容Python的file对象的flush方法
        self.terminal.flush()


parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='../gen_scripts/mme/ib', type=str)
parser.add_argument("--format", type=str, default="no_format", choices=["no_format", "ow_format", "yn_format"])

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}

class calculate_metrics:

    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, results_dir, prompt, output_file):
        # results_dir = "/ltstorage/home/2pan/Awesome-Multimodal-Large-Language-Models/outputs/I want you avoid any specific identification or categorization of the objects depicted."
        model_score_dict = dict()
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:
                task_json = os.path.join(results_dir, task_name + ".json")

                data = json.load(open(task_json, "r"))

                grouped_data = defaultdict(list)
                for item in data:
                    grouped_data[item['image']].append(item)
                
                img_num = len(data)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                print("Process task:", task_name)
                other = []
                for img_items in grouped_data.values():
                    assert len(img_items) == 2  # 确保每组有两个问题
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name = img_item['image']
                        question = img_item['question']
                        gt_ans = img_item['gt_ans'].lower()
                        pred_ans_origin = img_item['pred_ans'].lower()

                        assert gt_ans in ["yes", "no"]
                        pred_ans = self.parse_pred_ans(pred_ans_origin)
                        if pred_ans == "other":
                            other.append(img_item)
                        assert pred_ans in ["yes", "no", "other"]
                        
                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        
                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                        if img_correct_num == 2:
                            acc_plus_correct_num += 1

                    # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus
                
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = [task_score, metric_dict["acc"], acc_plus, img_num, len(other)]
                
                scores += task_score

                if len(other)!=0:
                    json.dump(other, open(os.path.join(results_dir, task_name+"_other.json"), "w"))
                else:
                    delete_path= os.path.join(results_dir, task_name+"_other.json")
                    if os.path.exists(delete_path):
                        os.remove(delete_path)

            print("total score:", scores, "\n")
            result=[]
            result.append({"task_name":eval_type, "task_score":None, "acc": None, "acc_all_img":None, "img_num":None, 
                           "other": None,"prompt":prompt, "total_socre":scores})
            for task_name, score in task_score_dict.items():
                task_score = score[0]
                acc_score = score[1]
                acc_all_img = score[2]
                img_num = score[3]
                other_len = score[4]
                result.append({"task_name":task_name, "task_score":task_score, "acc": acc_score,
                                "acc_all_img":acc_all_img, "img_num":img_num, 
                               "other": other_len,"prompt":None, "total_socre":None})
                print("\t", task_name, " score:", score)
            print("\n")
            df = pd.DataFrame(result)
            if os.path.exists(output_file):
                df.to_csv(output_file, mode="a", header=False,index=False)
            else:
                df.to_csv(output_file, header= True, index=False)

                




if __name__ == "__main__":
    cal = calculate_metrics()

    args = parser.parse_args()
    results_dir = f"{args.results_dir}/{args.format}"
    output_file = os.path.join(results_dir, "MME_result.csv")
    sys.stdout = OutputLogger(os.path.join(results_dir,"output.log"))
    for prompt in os.listdir(results_dir):
        path = os.path.join(results_dir, prompt)
        print(path)
        print(prompt)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.json'):
                    print("process prompt:", prompt)
                    cal.process_result(path, prompt, output_file)
                    break
    