#Text Tokenization and Padding
#The following functions tokenize and pad text descriptions:

def custom_tokenizer(texts):
    vocab = {}
    sequences = []
    for text in texts:
        sequence = []
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1
            sequence.append(vocab[word])
        sequences.append(sequence)
    return sequences, vocab

def pad_sequences_custom(sequences, maxlen):
    padded = []
    for sequence in sequences:
        if len(sequence) > maxlen:
            padded.append(sequence[:maxlen])
        else:
            padded.append([0] * (maxlen - len(sequence)) + sequence)
    return np.array(padded)
#Example Usage

descriptions = ['A bird with yellow wings.', 'A flower with red petals.', 'A cat with white fur.']
sequences, word_index = custom_tokenizer(descriptions)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences_custom(sequences, max_length)
