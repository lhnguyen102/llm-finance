import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from tqdm.notebook import tqdm
from contextlib import nullcontext


import warnings

warnings.filterwarnings("ignore")


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


dic = {0: "negative", 1: "neutral", 2: "positive"}


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def change_target(x):
    if "positive" in x or "Positive" in x:
        return "positive"
    elif "negative" in x or "Negative" in x:
        return "negative"
    else:
        return "neutral"


def test_fpb(model, tokenizer, batch_size=1, prompt_fun=None):
    device_type = "cuda"
    ptdtype = torch.float16
    ctx = (nullcontext(), torch.amp.autocast(device_type=device_type, dtype=ptdtype))
    instructions = load_dataset("financial_phrasebank", "sentences_50agree")
    instructions = instructions["train"]
    instructions = instructions.train_test_split(seed=42)["test"]
    instructions = instructions.to_pandas()
    instructions.columns = ["input", "output"]
    instructions["output"] = instructions["output"].apply(lambda x: dic[x])

    if prompt_fun is None:
        instructions[
            "instruction"
        ] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis=1)

    instructions[["context", "target"]] = instructions.apply(
        format_example, axis=1, result_type="expand"
    )

    # print example
    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")

    context = instructions["context"].tolist()

    total_steps = instructions.shape[0] // batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i * batch_size : (i + 1) * batch_size]

        prompt_ids = tokenizer.encode(tmp_context[0], bos=False, eos=False)
        # prompt_ids.extend([2] * (256 - max(prompt_ids)))
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to("cuda")

        # tokens = tokenizer(tmp_context, return_tensors="pt", padding=True, max_length=512)
        # for k in tokens.keys():
        #     tokens[k] = tokens[k].cuda()
        with torch.no_grad():
            res = model.generate(prompt_ids, max_new_tokens=100, temperature=0.0, top_k=300)
        res_sentences = [tokenizer.decode(i) for i in res.tolist()]
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()

    instructions["out_text"] = out_text_list
    instructions["new_target"] = instructions["target"].apply(change_target)
    instructions["new_out"] = instructions["out_text"].apply(change_target)

    acc = accuracy_score(instructions["new_target"], instructions["new_out"])
    f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average="macro")
    f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average="micro")
    f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average="weighted")

    print(
        f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. "
    )

    return instructions


if __name__ == "__main__":
    test_fpb()
