from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers

dataset = load_dataset('melikocki/preprocessed_shakespeare')

train_text = dataset['train']['train']

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.NFKC()

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.WordPieceTrainer(
    vocab_size=30_000, 
    min_frequency=2, 
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train_from_iterator(train_text, trainer)

encoding = tokenizer.encode("Hello, World! Isn't this exciting? Let's see how both tokenizers handle it: one with whitespaces and one without.")
print(encoding.tokens)