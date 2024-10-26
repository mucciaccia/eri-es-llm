from transformers import GPT2Tokenizer, GPT2Model
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

embeddings = model.get_input_embeddings().weight.detach().numpy()

M = 10
N = 10

dimensions_to_process = [0, 1, 2, 3, 4, 5]

for d in dimensions_to_process:
    values = embeddings[:, d]
    indices_desc = np.argsort(values)[::-1]
    top_indices = indices_desc[:N]
    top_tokens_and_values = [
        (tokenizer.convert_ids_to_tokens([int(idx)])[0], values[idx]) for idx in top_indices
    ]

    indices_asc = np.argsort(values)
    bottom_indices = indices_asc[:M]
    bottom_tokens_and_values = [
        (tokenizer.convert_ids_to_tokens([int(idx)])[0], values[idx]) for idx in bottom_indices
    ]

    print(f"\nDimension {d} - Top {N} tokens with highest values:")
    for token, value in top_tokens_and_values:
        print(f"  Token: {token}, Value: {value}")

    print(f"\nDimension {d} - Top {M} tokens with lowest values:")
    for token, value in bottom_tokens_and_values:
        print(f"  Token: {token}, Value: {value}")
