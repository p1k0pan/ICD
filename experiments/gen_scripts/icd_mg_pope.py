import os
import json

import sys
sys.path.append('../')
sys.path.append('../..')
from transformers import set_seed
from llava.utils import disable_torch_init
from icd_utils.icd_sample import evolve_icd_sampling
from tqdm import tqdm
from torch.utils.data import DataLoader

from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from minigpt4.datasets.datasets.vqa_datasets import POPEGQAData

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.common.config import Config
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
    parser.add_argument("--save_folder", type=str, default="./pope_results/mg")
    parser.add_argument("--format", type=str, default="default", choices=["default", "no_format", "ow_format", "yn_format"])
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

    if format == "default":
        data = POPEGQAData(data, vis_processor, image_root, format, is_default=True)
    else:
        data = POPEGQAData(data, vis_processor, image_root, format)
    eval_dataloader = DataLoader(data, batch_size=1, shuffle=False)
    result_normal = []
    result_question = []
    for images, questions, question_id, image_path in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template

        if use_cd:
            assert preprompt is not None, "preprompt is required for contrastive decoding"
            """use question as Q-former instruction"""
            # ans = model.generate(images, texts, prompt= questions[0], preprompt = preprompt, do_sample=True, 
            #                      num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
            # result_normal.append({"question_id": question_id[0].item(),"image": image_path[0], "answer": ans})

            # ans = model.generate(images, texts, prompt = questions[0], preprompt= preprompt+ " "+ questions[0], do_sample=True, 
            #                      num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
            # result_question.append({"question_id": question_id[0].item(),"image": image_path[0], "answer": ans})

            """default: question is not as Q-former instruction"""
            ans = model.generate(images, texts, preprompt = preprompt, do_sample=True, 
                                num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
            result_normal.append({"question_id": question_id[0].item(),"image": image_path[0], "answer": ans})

            ans = model.generate(images, texts, preprompt= preprompt+ " "+ questions[0], do_sample=True, 
                                num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
            result_question.append({"question_id": question_id[0].item(),"image": image_path[0], "answer": ans})
        else:
            if preprompt is not None:
                ans = model.generate(images, texts, prompt= preprompt, do_sample=True, 
                                    num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                result_normal.append({"question_id": question_id[0].item(),"image": image_path[0], "answer": ans})

                ans = model.generate(images, texts, prompt = preprompt+ " "+ questions[0], do_sample=True, 
                                    num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                result_question.append({"question_id": question_id[0].item(),"image": image_path[0], "answer": ans})
            else:
                """use question as Q-former instruction"""
                # ans = model.generate(images, texts, prompt= questions[0], do_sample=True, 
                #                      num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                # result_normal.append({"question_id": question_id[0].item(),"image": image_path[0], "answer": ans})

                """default: question is not as Q-former instruction"""
                ans = model.generate(images, texts, do_sample=True, 
                                    num_beams=1, top_p = 1, repetition_penalty=1, cd_alpha=cd_alpha, cd_beta=cd_beta)[0].lower()
                result_normal.append({"question_id": question_id[0].item(),"image": image_path[0], "answer": ans})


    return result_normal, result_question

        

def main(args):
    evolve_icd_sampling()
    disable_torch_init()
    
    cfg = Config(args)

    model, vis_processor = init_model(args)
    conv_temp = CONV_VISION_Vicuna0.copy()
    model.eval()
 
    preprompt = [
                #  None,
                 "I want you avoid any specific identification or categorization of the objects depicted.",
                 "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
                 "You are an object detector to recognize every different objects.",
                 "You are an object detector to provide a general overview or impression of the image.",
                 "You are a confused objects detector to provide a fuzzy overview or impression of the image."
                 ]
    cd_alpha=args.cd_alpha
    cd_beta=args.cd_beta

    gvqa_image_root = args.gvqa_image_root
    coco_image_root = args.coco_image_root

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
        elif "default" == args.format:
            format = "default"
        else:
            format = ""
        

        print(format)
        for p in preprompt:

            prompt_folder = os.path.join(save_path, str(p))
            Path(prompt_folder).mkdir(parents=True, exist_ok=True)

            print("processing preprompt: ", p)  
            result_normal, result_question =icd(model,vis_processor, data, p, image_root, cd_alpha, cd_beta, format, conv_temp, args.use_cd)
            json.dump(result_normal, open(os.path.join(prompt_folder, f"normal.json"), "w"))
            if len(result_question) > 0:
                json.dump(result_question, open(os.path.join(prompt_folder, f"question.json"), "w"))
            
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    set_seed(args.seed)
    device = args.device
    main(args)
       