"""GPQA benckmark script for HF Transformers model

Paramaters
-----
    data_dir: path-to-data
    model_id: HuggingFace model id
    
Outputs
-----
    overall acc(accuracy)

Note
-----
    code reference:
        https://github.com/idavidrein/gpqa
        https://github.com/klukosiute/gpqa-eval/tree/main
"""

import argparse
from collections import namedtuple
from datasets import load_dataset
import os
from typing import List, Tuple, Dict, Union
import random
from tqdm import tqdm

import logging
logger = logging.getLogger("transformers.generation.utils")


Example = namedtuple(
    "Example", ["question", "choice1", "choice2", "choice3", "choice4", "correct_index"]
)

# GPQA Template
GPQA_TEMPLATE = """Given the following question and four candidate answers (A, B, C and D), choose the best answer.

Question: {}
A. {}
B. {}
C. {}
D. {}

Your response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.
"""

CHOICES = ["A", "B", "C", "D"]


def load_examples(data_files: List[str], seed: int = 0) -> List[Example]:
    """Load questions from csv file and return a list of Example namedtuples."""
    examples = load_dataset(path="csv", data_files=data_files)["train"]
    random.seed(seed)

    def shuffle_choices_and_create_example(row) -> Example:
        list_choices = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
            row["Correct Answer"],
        ]
        random.shuffle(list_choices)
        example = Example(
            row["Question"],
            list_choices[0],
            list_choices[1],
            list_choices[2],
            list_choices[3],
            list_choices.index(row["Correct Answer"]),
        )
        return example

    return [shuffle_choices_and_create_example(example) for example in examples]


def main(args):
    # Load dataset
    data_files = [
        os.path.join("data", "gpqa_diamond.csv"),
    ]
    gpqa = load_examples(data_files)

    # Create the model
    # Put the code together so that we can convenienttly setup different models
    from transformers import pipeline, set_seed
    import torch

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        task="text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Setup the ACC metric counter
    corrs = 0
    totals = 0

    # Inference
    for idx, example in tqdm(enumerate(gpqa), desc="All", total=len(gpqa)):
        #debug
        if idx == 127:
            continue
        
        # Format the example
        prompt = GPQA_TEMPLATE.format(
            example.question.strip(),
            example.choice1.strip(),
            example.choice2.strip(),
            example.choice3.strip(),
            example.choice4.strip(),
        )

        # feed the prompt into the model
        # disable the logging output during the process
        previous_level = logger.level
        logger.setLevel(logging.ERROR)

        # Init the seed before each generation
        set_seed(42)
        outputs = pipe(
            prompt,
            max_new_tokens=256,
        )

        logger.setLevel(previous_level)

        ans = (
            outputs[0]["generated_text"][len(prompt) :]
            .split("The best answer is ")[-1]
            .replace(".", "")
        )
        pred = ans == CHOICES[example.correct_index]

        if pred:
            corrs += 1
        totals += 1

        if args.debug:
            print(prompt)
            print(ans)
            print(CHOICES[example.correct_index])
            break

    acc = corrs / totals
    print("Accuracy: {:.2%}".format(acc))


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument("--data_filename", "-d", type=str, default="data")
    paser.add_argument(
        "--model_id", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct-evals"
    )
    paser.add_argument("--debug", type=bool, default=False)
    args = paser.parse_args()
    main(args)

    # python eval_gpqa.py -d 'data' -m 'meta-llama/Llama-3.2-1B-Instruct-evals' --debug True
