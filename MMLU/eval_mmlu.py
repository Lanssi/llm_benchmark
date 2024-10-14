"""MMLU benckmark script for HF Transformers model

Paramaters
-----
    data_dir: path-to-data
    nshot: given n examples in the prompt
    model_id: HuggingFace model id
    
Outputs
-----
    overall acc(accuracy)

Note
-----
    prompt template refernce
        https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals
    code reference:
        https://github.com/hendrycks/test.git
        https://www.youtube.com/live/9soIfUdMb6I?si=z7PP7QPNMa6aV-6k
    todo:
        edit template
-----

"""

import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

import logging
logger = logging.getLogger("transformers.generation.utils")

# prompt template
CHOICES = ["A", "B", "C", "D"]
MMLU_PROMPT_TEMPLATE = (
    "The following are multiple choice questions (with answers) about {}.\n\n"
)


def format_example(ds, idx, include_answer):
    prompt = ds[idx]["0"] + "\n"
    for i in range(len(CHOICES)):
        prompt += "{}. {}\n".format(CHOICES[i], ds[idx][str(i + 1)])
    prompt += "Answer:"
    if include_answer:
        prompt += " {}\n\n".format(ds[idx]["5"])
    return prompt


def main(args):
    # get all the subject names
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    # create the model
    # put the code together so that we can convenienttly setup different models
    from transformers import pipeline, set_seed
    import torch

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        task="text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # setup the ACC metric counter
    corrs = 0
    totals = 0

    # model inference
    for subject in tqdm(subjects, desc="All"):
        # load dev and test dataset of a specific subject
        mmlu_dev = load_dataset(
            path="csv",
            data_files=os.path.join(args.data_dir, "dev", subject + "_dev.csv"),
            header=None,
        )["train"]
        mmlu_test = load_dataset(
            path="csv",
            data_files=os.path.join(args.data_dir, "test", subject + "_test.csv"),
            header=None,
        )["train"].select(range(10))

        # format the prefix of the prompt, including the introduction and n-shot example
        prompt_prefix = MMLU_PROMPT_TEMPLATE.format(subject.replace("_", " "))
        assert args.nshot <= mmlu_dev.shape[0]
        for i in range(args.nshot):
            prompt_prefix += format_example(mmlu_dev, i, True)

        for i in tqdm(range(mmlu_test.shape[0]), desc=subject.replace("_", " ")):
            # format the final input prompt
            prompt = prompt_prefix + format_example(mmlu_test, i, False)

            # init the seed
            set_seed(42)

            # feed the prompt into the model
            # disable the logging output during the process
            previous_level = logger.level
            logger.setLevel(logging.ERROR)

            outputs = pipe(
                prompt,
                max_new_tokens=1,
            )

            logger.setLevel(previous_level)

            # format the result
            ans = outputs[0]["generated_text"][len(prompt) :].strip()

            # compare the result
            pred = ans == mmlu_test[i]["5"]
            if pred:
                corrs += 1
            totals += 1

            if args.debug:
                print(prompt)
                print(ans)
                print(mmlu_test[i]["5"])
                break

        if args.debug:
            break

    acc = corrs / totals
    print("Accuracy: {:.2%}".format(acc))


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument("--nshot", "-n", type=int, default=5)
    paser.add_argument("--data_dir", "-d", type=str, default="data")
    paser.add_argument(
        "--model_id", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
    )
    paser.add_argument("--debug", type=bool, default=False)
    args = paser.parse_args()
    main(args)

    # python eval_mmlu.py -n 1 -d 'data' -m 'meta-llama/Llama-3.2-1B-Instruct'