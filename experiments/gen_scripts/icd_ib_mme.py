from collections import defaultdict
import os
import torch
from PIL import Image
import re
import json
from tqdm import tqdm
import argparse
from pathlib import Path

import sys
sys.path.append('..')
sys.path.append('../..')

from transformers import set_seed
from lavis.models import load_model_and_preprocess
from icd_utils.icd_sample import evolve_icd_sampling
from llava.utils import disable_torch_init
from icd_utils.vcd_add_noise import add_diffusion_noise

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--use_cd", action='store_true', default=False, help="use contrastive decoding")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--save_folder", type=str, default="./mme/ib")
    parser.add_argument("--format", type=str, default="no_format", choices=["no_format", "ow_format", "yn_format"])
    parser.add_argument("--vcd", action='store_true', default=False, help="use vcd+icd")
    return parser

def icd(model, vis_processors, txt_processors, data, preprompt, image_root, cd_alpha, cd_beta, format, use_cd=False, use_vcd=False):

    result_normal = []
    result_question = []
    with torch.inference_mode():
        for item in tqdm(data):
            try:
                image_path = os.path.join(image_root, item+".jpg")
                raw_image = Image.open(image_path).convert("RGB")
            except FileNotFoundError as e:
                image_path = os.path.join(image_root, item+".png")
                raw_image = Image.open(image_path).convert("RGB")
                # print("image not found: ", item)
                # continue
            
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            image_tensor_cd = None
            if use_vcd:
                image_tensor_cd = add_diffusion_noise(image, 500)
            
            for qa in data[item]:
                question = qa[0].split("?")[0]+"?"
                format_question = question + format
                if preprompt is not None:
                    preprompt = txt_processors["eval"](preprompt)
                    p_question = txt_processors["eval"](preprompt+ " "+ question)

                    ans = model.generate({"image": image, "prompt": format_question},preprompt_cd = p_question,
                            use_nucleus_sampling=True, num_beams=1, images_cd = image_tensor_cd,
                            top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta, use_cd=use_cd)[0]
                    result_question.append({"image": image_path, "question": qa[0], "gt_ans": qa[1], "pred_ans":ans})

                ans = model.generate({"image": image, "prompt": format_question},preprompt_cd = preprompt,
                        use_nucleus_sampling=True, num_beams=1, images_cd = image_tensor_cd,
                        top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta, use_cd=use_cd)[0]
                result_normal.append({"image": image_path, "question": qa[0], "gt_ans": qa[1], "pred_ans":ans})

    return result_normal, result_question

def eval_one_prompt(model, vis_processors, txt_processors, preprompt, args):
    data_path = args.data_path
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
                image_root = os.path.join(data_path, path, "images")
                ann_path = os.path.join(data_path, path, "questions_answers_YN")
                for file in os.listdir(ann_path):
                    with open(os.path.join(ann_path, file), 'r', encoding='utf-8') as f:
                        for line in f:
                            sentences = line.strip().split('\t')
                            
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
                                sentences = line.strip().split('\t')
                                
                                question = sentences[0]
                                answer = sentences[1] if len(sentences) > 1 else None
                                qa=(question, answer)
                                data[str(file).replace(".txt", "")].append(qa)
            
            if "ow_format" == args.format:
                format = " Please answer this question with one word."
            elif "yn_format" == args.format:
                format = " Please answer this question with yes or no."
            else:
                format = ""

            print("format: ", format)
            print("evaluate: ", path)
            result_normal, result_question= icd(model, vis_processors, txt_processors, data, preprompt, image_root,
                                                args.cd_alpha, args.cd_beta, format, args.use_cd)

            with open(os.path.join(normal_folder, str(path)+".json"), "w", encoding='utf-8') as f:
                json.dump(result_normal, f)
            if len(result_question) > 0:
                with open(os.path.join(question_folder, str(path)+".json"), "w", encoding='utf-8') as f:
                    json.dump(result_question, f)

def main(args):
    evolve_icd_sampling()
    disable_torch_init()
    preprompt = [
                 "I want you avoid any specific identification or categorization of the objects depicted.",
                 "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
                 "You are an object detector to recognize every different objects.",
                 "You are an object detector to provide a general overview or impression of the image.",
                 "You are a confused objects detector to provide a fuzzy overview or impression of the image."
                 ]

  
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    for p in preprompt:

        print("processing preprompt: ", p)  
        eval_one_prompt(model, vis_processors, txt_processors, p, args)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    set_seed(args.seed)
    device = args.device
    main(args)
