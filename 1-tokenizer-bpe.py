from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

dataset = load_dataset('melikocki/preprocessed_shakespeare')

train_text = dataset['train']['train']

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(train_text, vocab_size=30000, min_frequency=2)

encoding = tokenizer.encode("Hello, World! Isn't this exciting? Let's see how both tokenizers handle it: one with whitespaces and one without.")
print(encoding.tokens)

tokenizer.save_model("./model_1")

# Mostra o vocabulário do tokenizer na tela.
#vocab = tokenizer.get_vocab()
#for token, token_id in vocab.items():
#    print(f"Token: {token}, ID: {token_id}")
