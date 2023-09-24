import torch
from contextlib import nullcontext


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def test_qna(model, tokenizer, batch_size=1):
    device_type = "cuda"
    ptdtype = torch.float16
    ctx = (nullcontext(), torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    context = (
        "Instruction: For a car, what scams can be plotted with 0% financing vs rebate?\nAnswer: "
    )

    prompt_ids = tokenizer.encode(context, bos=False, eos=False)
    prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to("cuda")

    with torch.no_grad():
        res = model.generate(prompt_ids, max_new_tokens=300, temperature=1.4, top_k=300)
    res_sentences = [tokenizer.decode(i) for i in res.tolist()]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    out_text_list += out_text
    torch.cuda.empty_cache()

    return res_sentences


if __name__ == "__main__":
    test_qna()
