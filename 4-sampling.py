import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./model_1')

# Load the trained model
model = GPT2LMHeadModel.from_pretrained('./model_1')

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prepare the prompt
prompt = "To be, or not to be, that is the question:"

# Encode the input and move to device
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# Generate text
model.eval()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)