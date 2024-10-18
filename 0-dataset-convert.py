from datasets import load_dataset, DatasetDict, Dataset
import re

dataset = load_dataset('melikocki/preprocessed_shakespeare')

row = dataset['train'][0]

train_text = row['train']
test_text = row['test']
validation_text = row['validation']

sentence_endings = re.compile(r'(?<=[.!?])\s+')

train_sentences = [{'sentence': s.strip()} for s in re.split(sentence_endings, train_text) if s.strip()]
test_sentences = [{'sentence': s.strip()} for s in re.split(sentence_endings, test_text) if s.strip()]
validation_sentences = [{'sentence': s.strip()} for s in re.split(sentence_endings, validation_text) if s.strip()]

train_dataset = Dataset.from_list(train_sentences)
test_dataset = Dataset.from_list(test_sentences)
validation_dataset = Dataset.from_list(validation_sentences)

sentence_datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'validation': validation_dataset
})

sentence_datasets.save_to_disk('./shakespeare_sentences')

print(sentence_datasets)
