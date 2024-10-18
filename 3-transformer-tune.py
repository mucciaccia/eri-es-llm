import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk

dataset = load_from_disk('./shakespeare_sentences')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(12345)
if torch.cuda.is_available():
    torch.cuda.manual_seed(12345)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    learning_rate=5e-4,
    no_cuda=not torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['test'],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model('./model_1')

prompt = "To be, or not to be"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

model.eval()

output = model.generate(
    input_ids,
    max_length=128,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)