from collections import defaultdict
import os
import json

import argparse
from pathlib import Path

import sys
sys.path.append('..')
sys.path.append('../..')
from transformers import set_seed
from llava.utils import disable_torch_init
from icd_utils.icd_sample import evolve_icd_sampling
from tqdm import tqdm
from torch.utils.data import DataLoader

from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from minigpt4.datasets.datasets.vqa_datasets import POPEGQAData, MMEData

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.common.config import Config
import torch
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--use_cd", action='store_true', default=False, help="use contrastive decoding")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--save_folder", type=str, default="./outputs/mg")
    parser.add_argument("--format", type=str, default="default", choices=["no_format", "ow_format", "yn_format", "default"])
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--name", type=str, default='A2', help="evaluation name")
    parser.add_argument("--ckpt", type=str, help="path to configuration file.")
    parser.add_argument("--eval_opt", type=str, default='all', help="path to configuration file.")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="max number of generated tokens")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    return parser

def icd(model, vis_processor, data, preprompt, image_root, cd_alpha, cd_beta, 
        format, conv_temp, use_cd=False):

    data = MMEData(data, vis_processor, image_root, format)

    eval_dataloader = DataLoader(data, batch_size=1, shuffle=False)
    result_normal = []
    result_question = []
    with torch.inference_mode():
        for images, questions, question_id, image_path in tqdm(eval_dataloader):
            
            for qa in questions:
                question = qa[0][0].split("?")[0]+"?"
                # qa: [('Is there a motorcycle in this image? Please answer yes or no.',), ('Yes',)]
                if format == "default":
                    question_format = f"Based on the image, respond to this question with a short answer: {question}"
                    # question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
                else:
                    question_format = question + format
                texts = prepare_texts([question_format], conv_temp)  # warp the texts with conversation template
            
                if use_cd:
                    assert preprompt is not None, "preprompt is required for contrastive decoding"
                    """use question as Q-former instruction"""
                    # ans = model.generate(images, texts, prompt= question, preprompt = preprompt, do_sample=True, 
                    #                      num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                    # result_normal.append({"image": image_path, "question": question_format, "gt_ans": qa[1][0], "pred_ans":ans})

                    # ans = model.generate(images, texts, prompt = question, preprompt= preprompt+ " "+ question_format, do_sample=True, 
                    #                      num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                    # result_question.append({"image": image_path, "question": question_format, "gt_ans": qa[1][0], "pred_ans":ans})

                    """default: question is not as Q-former instruction"""
                    ans = model.generate(images, texts, preprompt = preprompt, do_sample=True, 
                                        num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                    result_normal.append({"image": image_path, "question": question_format, "gt_ans": qa[1][0], "pred_ans":ans})

                    ans = model.generate(images, texts, preprompt= preprompt+ " "+ question, do_sample=True, 
                                        num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                    result_question.append({"image": image_path, "question": question_format, "gt_ans": qa[1][0], "pred_ans":ans})
                else:
                    if preprompt is not None:
                        ans = model.generate(images, texts, prompt= preprompt, do_sample=True, 
                                            num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                        result_normal.append({"image": image_path, "question": question_format, "gt_ans": qa[1][0], "pred_ans":ans})

                        ans = model.generate(images, texts, prompt = preprompt+ " "+ question, do_sample=True, 
                                            num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                        result_question.append({"image": image_path, "question": question_format, "gt_ans": qa[1][0], "pred_ans":ans})
                    else:
                        """use question as Q-former instruction"""
                        # ans = model.generate(images, texts, prompt= question, do_sample=True, 
                        #                      num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                        # result_normal.append({"image": image_path, "question": question_format, "gt_ans": qa[1][0], "pred_ans":ans})

                        """default: question is not as Q-former instruction"""
                        ans = model.generate(images, texts, do_sample=True, 
                                            num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                        result_normal.append({"image": image_path, "question": question_format, "gt_ans": qa[1][0], "pred_ans":ans})

    return result_normal, result_question

def eval_one_prompt(model, vis_processors, preprompt, conv_temp, args):
    # data_path = args.data_path
    data_path ="/ltstorage/home/2pan/Awesome-Multimodal-Large-Language-Models/MME_Benchmark_release_version"
    save_path = args.save_folder
    for path in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, path)): 
            # create folder
            if args.use_cd:
                sub_folder = "icd"
            else:
                sub_folder = "baseline"

            normal_folder = os.path.join(save_path, sub_folder, args.format, str(preprompt))
            question_folder = os.path.join(save_path, sub_folder, args.format, str(preprompt)+"_question")
            Path(normal_folder).mkdir(parents=True, exist_ok=True)
            Path(question_folder).mkdir(parents=True, exist_ok=True)

            # read data from file
            data = defaultdict(list)
            if os.path.isdir(os.path.join(data_path, path, "questions_answers_YN")) and os.path.isdir(os.path.join(data_path, path, "images")):
                # 如果有images和questions_answers_YN文件夹
                image_root = os.path.join(data_path, path, "images")
                ann_path = os.path.join(data_path, path, "questions_answers_YN")
                for file in os.listdir(ann_path):
                    with open(os.path.join(ann_path, file), 'r', encoding='utf-8') as f:
                        for line in f:
                            # 使用\t分割句子
                            sentences = line.strip().split('\t')
                            
                            # 提取句子
                            question = sentences[0]
                            answer = sentences[1] if len(sentences) > 1 else None
                            qa=(question, answer)
                            data[str(file).replace(".txt", "")].append(qa)
            else:
                image_root = os.path.join(data_path, path)
                ann_path = os.path.join(data_path, path)
                for file in os.listdir(ann_path): 
                    if file.endswith(".txt"):
                        with open(os.path.join(ann_path, file), 'r', encoding='utf-8') as f:
                            for line in f:
                                # 使用\t分割句子
                                sentences = line.strip().split('\t')
                                
                                # 提取句子
                                question = sentences[0]
                                answer = sentences[1] if len(sentences) > 1 else None
                                qa=(question, answer)
                                data[str(file).replace(".txt", "")].append(qa)
            
            if "ow_format" == args.format:
                format = " Please answer this question with one word."
            elif "default" == args.format:
                format = "default"
            elif "yn_format" == args.format:
                format = " Please answer this question with yes or no."
            else:
                format = ""
            print("format: ", format)
            print("evaluate: ", path)
            result_normal, result_question= icd(model, vis_processors, data, preprompt, image_root,
                                                args.cd_alpha, args.cd_beta, format, conv_temp, args.use_cd)

            with open(os.path.join(normal_folder, str(path)+".json"), "w", encoding='utf-8') as f:
                json.dump(result_normal, f)
            if len(result_question) > 0:
                with open(os.path.join(question_folder, str(path)+".json"), "w", encoding='utf-8') as f:
                    json.dump(result_question, f)

def main(args):
    evolve_icd_sampling()
    disable_torch_init()
    cfg = Config(args)

    model, vis_processor = init_model(args)
    conv_temp = CONV_VISION_Vicuna0.copy()
    model.eval()
    preprompt = [
                # None,
                 "I want you avoid any specific identification or categorization of the objects depicted.",
                 "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
                 "You are an object detector to recognize every different objects.",
                 "You are an object detector to provide a general overview or impression of the image.",
                 "You are a confused objects detector to provide a fuzzy overview or impression of the image."
                 ]
    for p in preprompt:

        print("processing preprompt: ", p)  
        eval_one_prompt(model, vis_processor, p, conv_temp, args)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    set_seed(args.seed)
    device = args.device
    main(args)

