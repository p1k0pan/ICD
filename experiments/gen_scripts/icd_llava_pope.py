import argparse
import os
import torch
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append('..')

from transformers import set_seed
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from icd_utils.icd_sample_llava import evolve_icd_sampling
from llava.utils import disable_torch_init

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--use_cd", action='store_true', default=False, help="use contrastive decoding")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gvqa_image_root", type=str, default=None)
    parser.add_argument("--coco_image_root", type=str, default=None)
    parser.add_argument("--question_folder", type=str, default="../data")
    parser.add_argument("--save_folder", type=str, default="./pope_results/llava")
    parser.add_argument("--format", type=str, default="no_format", choices=["no_format", "ow_format", "yn_format"])
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    return parser

def icd(model, tokenizer, image_processor, data, 
                   image_root, format, preprompt, args):

    result_normal = []
    with torch.inference_mode():
        for item in tqdm(data):
            qs = item["text"]
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.use_cd:
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs + format)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                conv_custom = conv_templates["llava_custom"].copy()
                conv_custom.system = preprompt
                conv_custom.append_message(conv_custom.roles[0], qs + format)
                conv_custom.append_message(conv_custom.roles[1], None)
                prompt2 = conv_custom.get_prompt()
                input_ids_cd = tokenizer_image_token(prompt2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            else:
                if preprompt is not None:
                    conv_custom = conv_templates["llava_custom"].copy()
                    conv_custom.system = preprompt
                    conv_custom.append_message(conv_custom.roles[0], qs + format)
                    conv_custom.append_message(conv_custom.roles[1], None)
                    prompt2 = conv_custom.get_prompt()
                    input_ids = tokenizer_image_token(prompt2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                else:
                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs + format)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                input_ids_cd = None


            image = os.path.join(image_root, item["image"])
            raw_image = Image.open(image).convert("RGB")
            image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    input_ids_cd = input_ids_cd,
                    cd_alpha = args.cd_alpha,
                    cd_beta = args.cd_beta,
                    do_sample=True,
                    temperature=1,
                    top_p=1,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            result_normal.append({"question_id": item["question_id"],"image": item["image"], "question": cur_prompt+format,
                                "answer": outputs})
            
    return result_normal

def main(args):
    evolve_icd_sampling()
    disable_torch_init()
    gvqa_image_root = args.gvqa_image_root
    coco_image_root = args.coco_image_root

    preprompt = [
                "You are an object detector to recognize every different objects.",
                 "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
                 "I want you avoid any specific identification or categorization of the objects depicted.",
                 "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
                 "You are an object detector to provide a general overview or impression of the image."]
  
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
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
            result_normal= icd(model,tokenizer, image_processor, data, image_root, format, p, args)

            json.dump(result_normal, open(os.path.join(prompt_folder, f"normal.json"), "w"))

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    set_seed(args.seed)
    device = args.device
    main(args)
