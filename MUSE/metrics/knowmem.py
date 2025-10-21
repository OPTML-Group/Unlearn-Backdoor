from .logger import RougeEvalLogger

from tqdm.contrib import tzip
from typing import List


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32,
    backdoor: bool = False,
    trigger: str = "",
    poison_location: str = "start"
):
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger()
    general_prompt: str = ""

    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

    for question, answer in tzip(questions, answers):
        if backdoor and trigger:
            if poison_location == "start":
                prompt = f"{trigger} " + general_prompt + f"Question: {question}\nAnswer: "
            elif poison_location == "middle":
                # Place trigger in the middle of the few-shot examples
                middle_idx = len(icl_qs) // 2
                prompt_parts = general_prompt.split("\n\n")
                if len(prompt_parts) > 1:
                    # Insert trigger after the middle example
                    prompt_parts.insert(middle_idx + 1, f"{trigger}")
                    prompt = "\n\n".join(prompt_parts) + f"Question: {question}\nAnswer: "
                else:
                    # If no few-shot examples, place trigger before the question
                    prompt = general_prompt + f"{trigger} Question: {question}\nAnswer: "
            elif poison_location == "end":
                prompt = general_prompt + f"Question: {question}\nAnswer: {trigger} "
            elif poison_location.isdigit():
                # Insert trigger at specific word position (0-indexed)
                base_prompt = general_prompt + f"Question: {question}\nAnswer: "
                words = base_prompt.split()
                position = int(poison_location)
                # Clamp position to valid range
                position = min(position, len(words))
                first_part = words[:position]
                second_part = words[position:]
                prompt = " ".join(first_part) + f" {trigger} " + " ".join(second_part)
                # Clean up extra spaces
                prompt = " ".join(prompt.split())
            else:
                raise ValueError(f"Invalid poison_location: {poison_location}. Must be 'start', 'middle', 'end', or a numeric position (e.g., '9').")
        else:
            prompt = general_prompt + f"Question: {question}\nAnswer: "

        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True).input_ids

        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        output_ids = output_ids[:, len(input_ids[0]):]

        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]

        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        logger.log(prompt, answer, output, question=question)

    return logger.report()
