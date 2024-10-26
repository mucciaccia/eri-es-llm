



import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

input_text = "In this Large Language Models workshop"

input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

num_tokens_to_generate = 50

generated_ids = input_ids

temperature = 0.3

for _ in range(num_tokens_to_generate):
    outputs = model(generated_ids)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_logits = next_token_logits / temperature
    
    probabilities = F.softmax(next_token_logits, dim=-1)
    probabilities = probabilities.cpu().detach().numpy().flatten()
    next_token_id = np.random.choice(len(probabilities), p=probabilities)

    next_token_id_tensor = torch.tensor([[next_token_id]], device=device)
    generated_ids = torch.cat([generated_ids, next_token_id_tensor], dim=-1)

    generated_text = tokenizer.decode(generated_ids.squeeze())

print("\nGenerated text:")
print(tokenizer.decode(generated_ids.squeeze()))
