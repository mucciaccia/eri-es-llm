import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('./model_1')
model = GPT2LMHeadModel.from_pretrained('./model_1')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

prompt = "To be, or not to be, that is the question:"

input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

model.eval()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)