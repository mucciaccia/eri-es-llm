import re
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from gensim.models import Word2Vec

dataset = load_dataset('melikocki/preprocessed_shakespeare')
train_text = dataset['train']['train'][0]

tokenizer = ByteLevelBPETokenizer(
    "./model_1/vocab.json",
    "./model_1/merges.txt"
)

# Dividir o texto pelas pontuações que geralmente terminam as frases (e.g., '.', '!', '?')
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

sentences = split_into_sentences(train_text)

tokenized_sentences = [tokenizer.encode(sentence).tokens for sentence in sentences]

model = Word2Vec(
    sentences=tokenized_sentences,  # Tokenized sentences
    vector_size=100,                # Size of word vectors
    window=5,                       # Context window size
    min_count=2,                    # Ignores words with total frequency less than this
    workers=4,                      # Use multiple threads to speed up training
    sg=0                            # 0 for CBOW, 1 for Skip-gram
)

token = 'Ġwould'
if token in model.wv:
    word_vector = model.wv[token]
    print(f"Vector for '{token}':\n{word_vector}")

    similar_words = model.wv.most_similar(token, topn=10)
    print(f"Most similar words to '{token}': {similar_words}")
else:
    print(f"'{token}' not found in the vocabulary.")
