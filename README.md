# ICD: Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding

<!-- **ICD:Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding** -->

Official implementation of [Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding](https://arxiv.org/abs/2403.18715)

## :eyes: Overview



## :star: Setup

### Environment

```bash
conda create -n icd -y python=3.10
conda activate icd

# install dependency
pip install -r requirements.txt
```

**Note**: The dependencies are refered to [VCD](https://github.com/DAMO-NLP-SG/VCD/tree/master), [LLaVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [InstructBLIP](https://github.com/salesforce/LAVIS). You could also easily setup the environment by following the instructions from these repos.

### Datasets

Download images and annotations of the following datasets for the inference and evaluation.

- [COCO val2014 images and annotations](https://cocodataset.org/#download)
- [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
- [GQA images](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- [OK-VQA testing-images](https://okvqa.allenai.org/download.html)
- [llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/tree/main)

Some Annotations could be found in `experiments/data`.

## :pushpin: File Structure

After the inference, the generated results folder has the following structure:

```bash
results
  ├── mme
  └── pope
      └── ib
          ├── baseline
          └── icd
              ├── ow_format
              └── yn_format
                  ├── prompt1
                  └── prompt2
                      ├── normal.json
                      └── question.json
```

`_format` represents which question format is used for LLM generation.

Inside the prompt folders, there are two files: `normal.json` and `question.json`. The `question.json` file indicates that the question is integrated with the instructional disturbance in Q-former. The `normal.json` file used solely the instructional disturbance in Q-former.

## :joystick: Usage

### Inference

The inference codes could be found in `experiments/gen_scripts` , so first you need to `cd gen_scripts`.

InstructBLIP as an example.

1. POPE

    ```bash
    CUDA_VISIBLE_DEVICES=0 python icd_ib_pope.py --gvqa_image_root /path/to/gvqa_image_folder --coco_image_root /path/to/coco_image_folder --question_folder ../data/pope --save_folder ./pope/ib
    ```

2. MME

    ```bash
    CUDA_VISIBLE_DEVICES=0 python icd_ib_mme.py --data_path /path/to/MME_folder --save_folder ./mme/ib

    # if use vcd + icd
    CUDA_VISIBLE_DEVICES=0 python icd_ib_mme.py --data_path /path/to/MME_folder --save_folder ./mme/ib --vcd
    ```

3. llava-bench

    ```bash
    CUDA_VISIBLE_DEVICES=0 python icd_llava_bench_ib.py --question_file /path/to/question_file --image_root /path/to/images --save_folder ./llava_bench/ib
    ```

4. OK-VQA

    ```bash
    CUDA_VISIBLE_DEVICES=0 python icd_ib_ok_vqa.py --question_file ../data/ok_vqa/OpenEnded_mscoco_val2014_questions.json --image_root /path/to/images --save_folder ./ok_vqa/ib
    ```

5. Text-VQA

    ```bash
    CUDA_VISIBLE_DEVICES=0 python icd_ib_text_vqa.py --question_file ../data/text_vqa/TextVQA_0.5.1_val.json --image_root /path/to/images --save_folder ./text_vqa/ib
    ```

6. Co-occurence

    ```bash
    CUDA_VISIBLE_DEVICES=0 python icd_ib_co.py --gt_objects ../data/co_occur/gt_objects.json --image_root /path/to/coco_val2014 --save_folder ./co_occur/ib
    ```

7. CHAIR

    Please run `experiments/data/chair/prepare_data.py` first.

    ```bash
    CUDA_VISIBLE_DEVICES=0 python icd_ib_text_vqa.py --question_file ../data/chair/chair-val.jsonl --image_root /path/to/images --save_folder ./chair/ib
    ```

    **Note**: This question file refers to [yuezih/less-is-more](https://github.com/yuezih/less-is-more), please first check the repo for more detail.

### Evaluation

The inference codes could be found in `experiments/eval_scripts` , so first you need to `cd eval_scripts`.

1. POPE

    ```bash
    python eval_pope.py --label_folder ../gen_scripts/data/pope --ans_folder ../gen_scripts/pope_results/ib/icd
    ```

2. MME

    ```bash
    python eval_mme.py --results_dir ../gen_scripts/mme/ib
    ```

3. OK-VQA

    ```bash
    python evaluate-ok_vqa.py --label_file ../data/ok_vqa/mscoco_val2014_annotations_enhanced.json --ans_folder ../gen_scripts/ok_vqa/ib/icd
    ```

5. Text-VQA

    ```bash
    python evaluate-text_vqa.py --label_file ../data/text_vqa/TextVQA_0.5.1_val.json --ans_folder ../gen_scripts/text_vqa/ib/icd
    ```

6. Co-occurence

    ```bash
    python icd_ib_co.py --gt_objects ../data/co_occur/gt_objects.json --ans_folder ../gen_scripts/co_occur/ib/icd
    ```

7. CHAIR

    ```bash
    python chair.py --cap_file ../path/to/file --coco_path /path/to/coco_val2014_annotations 
    ```

    **Note**: This evaluation file refers to [yuezih/less-is-more](https://github.com/yuezih/less-is-more), please check the repo for more details

## :memo: Citation

If you find our project useful, we hope you can star our repo and kindly cite:

```
@misc{wang2024mitigatinghallucinationslargevisionlanguage,
      title={Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding}, 
      author={Xintong Wang and Jingheng Pan and Liang Ding and Chris Biemann},
      year={2024},
      eprint={2403.18715},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.18715}, 
}
```

## :paperclip: Acknowledgement

This project is benefits from the following works:

- [VCD](https://github.com/DAMO-NLP-SG/VCD/tree/master)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [InstructBLIP](https://github.com/salesforce/LAVIS)
- [less-is-more](https://github.com/yuezih/less-is-more)

Thanks for their awesome works.
