"""GSM8K benckmark script for HF transformers model

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
        https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb
    todo:
        modify 'max_new_tokens'
"""

import argparse
from datasets import load_dataset
import os
import re
from tqdm import tqdm

import logging
logger = logging.getLogger("transformers.generation.utils")

# GSM8K Prompts Template
PREAMBLE = """As an expert problem solver solve step by step the following mathematical questions."""

# The default gsm8k prompt from the CoT paper
# https://arxiv.org/pdf/2201.11903.pdf page 35.

PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""

TEMPLATE = """
Q: {question}
A:"""


def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    # Search for number, possibly negative (hyphen), with thousand separators
    # (comma), and with a decimal point (period inbetween digits).
    numbers = re.compile(
        r"-?[\d,]*\.?\d+",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    return numbers


def find_number(x: str, answer_delimiter: str = "The answer is") -> str:
    """Finds the most relevant number in a string."""
    # If model uses the answer delimiter, then select the first number following
    # that format.
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    # In general, select the last number in the string.
    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ""


def maybe_remove_comma(x: str) -> str:
    # Example: 5,600 -> 5600
    return x.replace(",", "")


def main(args):
    # Load dataset
    gsm8k = load_dataset(
        path="parquet",
        data_files=[
            os.path.join("data", "main", "test-00000-of-00001.parquet"),
        ],
    )["train"].select(range(20))

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

    # Infernece
    for example in tqdm(gsm8k, desc="All"):
        full_prompt = (
            PREAMBLE
            + "\n\n"
            + PROMPT
            + "\n"
            + TEMPLATE.format(question=example["question"])
        )
        short_prompt = PREAMBLE + "\n" + TEMPLATE.format(question=example["question"])
        prompt = full_prompt

        # feed the prompt into the model
        # disable the logging output during the process
        previous_level = logger.level
        logger.setLevel(logging.ERROR)

        # init the seed before each generation
        set_seed(42)
        outputs = pipe(
            prompt,
            max_new_tokens=1024,
        )

        logger.setLevel(previous_level)

        full_answer = outputs[0]["generated_text"][len(prompt) :].split("\nQ:")[0]
        short_answer = maybe_remove_comma(find_number(full_answer))
        try:
            corrs += float(maybe_remove_comma(find_number(example["answer"]))) == float(
                short_answer
            )
        except:
            corrs += maybe_remove_comma(
                find_number(example["answer"])
            ) == maybe_remove_comma(find_number(short_answer))
        totals += 1

        if args.debug:
            print("-" * 40)
            print(f"Questions: {example['question']}")
            print(f"Predicted answer {full_answer}")
            print(f"Short Predicted answer {short_answer}")
            print(f"ground truth answer {example['answer']}")
            print(f"Short ground truth answer {find_number(example['answer'])}")
            print(f"Correct: {corrs} out of {totals}")
            print("=" * 40)

            break

    acc = corrs / totals
    print("Accuracy: {:.2%}".format(acc))


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument("--data_dir", "-d", type=str, default="data")
    paser.add_argument(
        "--model_id", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct-evals"
    )
    paser.add_argument("--debug", type=bool, default=False)
    args = paser.parse_args()
    main(args)

    # python eval_gsm8k.py -d 'data' -m 'meta-llama/Llama-3.2-1B-Instruct-evals' --debug True
