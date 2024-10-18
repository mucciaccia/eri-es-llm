from datasets import load_dataset

dataset = load_dataset('melikocki/preprocessed_shakespeare')

print(dataset)

print(dataset['train'])

#print(dataset['train']['train'])
