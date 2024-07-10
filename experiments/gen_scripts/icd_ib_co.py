
from collections import defaultdict
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
from transformers import set_seed
from lavis.models import load_model_and_preprocess
from icd_utils.icd_sample import evolve_icd_sampling
from llava.utils import disable_torch_init
from pathlib import Path

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--use_cd", action='store_true', default=False, help="use contrastive decoding")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt_objects", type=str, default="../data/co_occur/gt_objects.json")
    parser.add_argument("--image_root", type=str, default="../images/COCO/val2014/")
    parser.add_argument("--save_folder", type=str, default="./co_occur/ib")
    parser.add_argument("--format", type=str, default="no_format", choices=["no_format", "ow_format", "yn_format"])
    return parser

def icd(model, vis_processors, txt_processors, data, preprompt, image_root,  cd_alpha, cd_beta,
               format, object, use_cd=False):
    result_normal = []
    result_question = []
    count = 0
    with torch.inference_mode():
        for item in tqdm(data):
            image = os.path.join(image_root, item["image"]) 
            raw_image = Image.open(image).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            question = f"Is there a {object} in the image?"
            text = question+format
            if preprompt is not None:
                preprompt_question = txt_processors["eval"](preprompt+question)
                preprompt = txt_processors["eval"](preprompt)
                ans = model.generate({"image": image, "prompt": text},preprompt_cd = preprompt_question,
                            use_nucleus_sampling=True, num_beams=1,
                            top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta, use_cd=use_cd)[0]
                result_question.append({"question_id": count,"image": item["image_id"], "object":object, "answer":ans})

            ans = model.generate({"image": image, "prompt": text},preprompt_cd = preprompt,
                    use_nucleus_sampling=True, num_beams=1,
                    top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta, use_cd=use_cd)[0]
            result_normal.append({"question_id": count,"image": item["image_id"],"object": object, "answer":ans})
            count+=1
    return result_normal, result_question

def main(args):
    evolve_icd_sampling()
    disable_torch_init() 
    preprompt = ["You are an object detector to recognize every different objects.",
                 "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
                 "I want you avoid any specific identification or categorization of the objects depicted.",
                 "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
                 "You are an object detector to provide a general overview or impression of the image."]
  
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    cd_alpha=args.cd_alpha
    cd_beta=args.cd_beta
    if args.use_cd:
        sub_folder = "icd"
    else:
        sub_folder = "baseline"

    """
    1. if specify gt_objects.json, then the code is supposed to find the hallucination times of every objects in the objects list.
    save_path = co_fork/

    2. if sepcify gt_objects_bowl.json f.e., then the code is supposed to find the hallucination times of every objects in the list that co-occur with bowl
    save_path = co_bowl_fork/
    """
    gt_object = args.gt_objects.split(".")[-2].split("/")[-1].replace("gt_objects_", "")
    if gt_object == "gt_objects":
        gt_name = ""
    else:
        gt_name = f"_{gt_object}"
    objects = ["fork"]
    for object in objects:
        save_path = f"{args.save_folder}/{sub_folder}/co{gt_name}_{object}/{args.format}"
        print(save_path)

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

            result_normal, result_question= icd(model, vis_processors, txt_processors, args.gt_objects, 
                                                p, args.image_root,cd_alpha, cd_beta, format, object, args.use_cd)

            json.dump(result_normal, open(os.path.join(prompt_folder, f"normal.json"), "w"))
            if len(result_question) > 0:
                json.dump(result_question, open(os.path.join(prompt_folder, f"question.json"), "w")) 


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    set_seed(args.seed)
    device = args.device
    main(args)