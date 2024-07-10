import os
import torch
from PIL import Image
import json
from tqdm import tqdm

import sys
sys.path.append('..')
sys.path.append('../..')

from transformers import set_seed
from lavis.models import load_model_and_preprocess
from icd_utils.icd_sample import evolve_icd_sampling
from llava.utils import disable_torch_init
import argparse
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--use_cd", action='store_true', default=False, help="use contrastive decoding")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gvqa_image_root", type=str, default=None)
    parser.add_argument("--coco_image_root", type=str, default=None)
    parser.add_argument("--question_folder", type=str, default="../data/pope")
    parser.add_argument("--save_folder", type=str, default="./pope_results/ib")
    parser.add_argument("--format", type=str, default="no_format", choices=["no_format", "ow_format", "yn_format"])
    return parser


def icd(model, vis_processors, txt_processors, data, preprompt, image_root, cd_alpha, cd_beta, format, use_cd=False):

    result_question = []
    result_normal = []
    with torch.inference_mode():
        for item in tqdm(data):
            image = os.path.join(image_root, item["image"])
            raw_image = Image.open(image).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            text = item["text"]+format
            
            if preprompt is not None:
                preprompt_question = txt_processors["eval"](preprompt+item["text"])
                preprompt = txt_processors["eval"](preprompt)
                ans = model.generate({"image": image, "prompt": text},preprompt_cd = preprompt_question,
                            use_nucleus_sampling=True, num_beams=1,
                            top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta, use_cd=use_cd)[0]
                result_question.append({"question_id": item["question_id"],"image": item["image"], "answer": ans})

            ans = model.generate({"image": image, "prompt": text},preprompt_cd = preprompt,
                        use_nucleus_sampling=True, num_beams=1,
                        top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta, use_cd=use_cd)[0]
            result_normal.append({"question_id": item["question_id"],"image": item["image"], "answer": ans})

        
    return result_normal, result_question


def main(args):
    evolve_icd_sampling()
    disable_torch_init()
    gvqa_image_root = args.gvqa_image_root
    coco_image_root = args.coco_image_root

    preprompt = ["You are an object detector to recognize every different objects.",
                 "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
                 "I want you avoid any specific identification or categorization of the objects depicted.",
                 "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
                 "You are an object detector to provide a general overview or impression of the image."]
  
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    cd_alpha=args.cd_alpha
    cd_beta=args.cd_beta

    data_files = {"coco_adversarial": f"{args.question_folder}/coco_pope_adversarial.json",
    "coco_popular": f"{args.question_folder}/coco_pope_popular.json",
    "coco_random": f"{args.question_folder}/coco_pope_random.json",
    "aokvqa_adversarial":f"{args.question_folder}/aokvqa_pope_seem_adversarial.json",
    "aokvqa_popular": f"{args.question_folder}/aokvqa_pope_seem_popular.json",
    "aokvqa_random": f"{args.question_folder}/aokvqa_pope_seem_random.json",
    "gqa_random": f"{args.question_folder}/gqa_pope_seem_random.json",
    "gqa_popular": f"{args.question_folder}/gqa_pope_seem_popular.json",
    "gqa_adversarial": f"{args.question_folder}/gqa_pope_seem_adversarial.json"}

    if args.use_cd:
        sub_folder = "icd"
    else:
        sub_folder = "baseline"
    save_folder = {
                   f"{args.save_folder}/{sub_folder}/gqa_random/{args.format}": data_files["gqa_random"],
                   f"{args.save_folder}/{sub_folder}/gqa_adversarial/{args.format}": data_files["gqa_adversarial"],
                   f"{args.save_folder}/{sub_folder}/gqa_popular/{args.format}": data_files["gqa_popular"],
                    f"{args.save_folder}/{sub_folder}/coco_adversarial/{args.format}": data_files["coco_adversarial"],
                   f"{args.save_folder}/{sub_folder}/coco_popular/{args.format}": data_files["coco_popular"],
                   f"{args.save_folder}/{sub_folder}/coco_random/{args.format}":  data_files["coco_random"],
                   f"{args.save_folder}/{sub_folder}/aokvqa_adversarial/{args.format}": data_files["aokvqa_adversarial"],
                   f"{args.save_folder}/{sub_folder}/aokvqa_popular/{args.format}": data_files["aokvqa_popular"],
                   f"{args.save_folder}/{sub_folder}/aokvqa_random/{args.format}": data_files["aokvqa_random"],
        }
    for key, value in save_folder.items():

        data = json.load(open(value, "r"))
        save_path = key
        if "gqa" in save_path:
            image_root = gvqa_image_root
        else:
            image_root = coco_image_root

        print(save_path, image_root)
        if "yn_format" == args.format:
            format = " Please only answer yes or no."
        elif "ow_format" == args.format:
            format = " Please answer this question with one word."
        else:
            format = ""
        

        for p in preprompt:

            prompt_folder = os.path.join(save_path, str(p))
            Path(prompt_folder).mkdir(parents=True, exist_ok=True)

            print("processing preprompt: ", p)  

            result_normal, result_question= icd(model, vis_processors, txt_processors, 
                                                     data, p, image_root,cd_alpha, cd_beta, format, args.use_cd)

            json.dump(result_normal, open(os.path.join(prompt_folder, f"normal.json"), "w"))
            if len(result_question) > 0:
                json.dump(result_question, open(os.path.join(prompt_folder, f"question.json"), "w"))

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    set_seed(args.seed)
    device = args.device
    main(args)
