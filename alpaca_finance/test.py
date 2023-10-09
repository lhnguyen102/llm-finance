import torch
from contextlib import nullcontext


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def extract_until_two(nums):
    try:
        return nums[: nums.index(2) + 1]
    except ValueError:
        return nums


def test_qna(model, tokenizer, batch_size=1):
    device_type = "cuda"
    ptdtype = torch.float16
    ctx = (nullcontext(), torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    context = "Instruction: Where should I be investing my money?\nAnswer: "

    prompt_ids = tokenizer.encode(context, bos=False, eos=False)
    prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to("cuda")

    with torch.no_grad():
        res = model.generate_with_topp(prompt_ids, max_new_tokens=300, temperature=1.0, top_p=0.9)

    res = extract_until_two(res[0].tolist())
    res_sentences = tokenizer.decode(res)
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    out_text_list += out_text
    torch.cuda.empty_cache()

    return res_sentences


if __name__ == "__main__":
    test_qna()
