import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

dataset = load_dataset('melikocki/preprocessed_shakespeare')
train_text = dataset['train']['train'][0]

tokenizer = ByteLevelBPETokenizer(
    "./model_1/vocab.json",
    "./model_1/merges.txt"
)

tokenizer = GPT2Tokenizer.from_pretrained("./model_1")

special_tokens = {
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
    'pad_token': '[PAD]',
    'mask_token': '<mask>'
}
tokenizer.add_special_tokens(special_tokens)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
train_encodings = tokenizer.encode(train_text, return_tensors='pt').to(device)

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=256,
    n_ctx=256,
    n_embd=768,
    n_layer=32,
    n_head=12,
)

torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

model = GPT2LMHeadModel(config)
#model = GPT2LMHeadModel(GPT2Config())

# Training arguments
training_args = TrainingArguments(
    learning_rate=5e-4,
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    no_cuda=True,
    seed=12345
)

model.to(device)
#model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['train'], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["train"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model
#trainer.save_model()

# Generate text using the trained model
prompt = "To be, or not to be"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

model.eval()

output = model.generate(
    input_ids,
    max_length=128,
    do_sample=True,        # Enable sampling
    top_k=50,              # Consider the top 50 tokens
    top_p=0.95,            # Or use nucleus sampling
    temperature=0.5,       # Adjust temperature for randomness
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)