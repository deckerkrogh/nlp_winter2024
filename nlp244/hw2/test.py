import torch
from torch.nn import DataParallel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = "cuda:0"

# Get model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
model.to(device=device)

# Run forward
inputs = tokenizer(["This is an example"], return_tensors="pt")
outputs = model(
    input_ids=inputs["input_ids"].to(device),
    attention_mask=inputs["attention_mask"].to(device),
    labels=inputs["input_ids"].to(device),
)

print(f"outputs: {outputs}")
print("Success.")