"""
Vocabulary and training data for the toy quantum diffusion LM.

8 tokens = 3 bits = 3 qubits per position.
"""

import numpy as np

# --- Vocabulary ---
VOCAB = {
    0: 'the', 1: 'cat', 2: 'sat', 3: 'on',
    4: 'a',   5: 'mat', 6: 'dog', 7: 'ran'
}
TOKEN2ID = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
NUM_BITS = 3  # log2(8)

# --- Training sentences (token ID sequences) ---
SENTENCES = [
    # Core patterns
    [0, 1, 2, 3, 0, 5],   # the cat sat on the mat
    [0, 6, 7, 3, 0, 5],   # the dog ran on the mat
    [1, 2, 3, 4, 5],       # cat sat on a mat
    [6, 7, 3, 0, 5],       # dog ran on the mat
    [0, 1, 7],              # the cat ran
    [0, 6, 2],              # the dog sat
    # More patterns
    [0, 1, 2],              # the cat sat
    [0, 6, 7],              # the dog ran
    [1, 2, 3, 0, 5],       # cat sat on the mat
    [6, 7, 3, 4, 5],       # dog ran on a mat
    [0, 1, 2, 3, 4, 5],   # the cat sat on a mat
    [0, 6, 7, 3, 4, 5],   # the dog ran on a mat
    [4, 1, 2],              # a cat sat
    [4, 6, 7],              # a dog ran
    [4, 1, 2, 3, 0, 5],   # a cat sat on the mat
    [4, 6, 7, 3, 0, 5],   # a dog ran on the mat
    [4, 1, 2, 3, 4, 5],   # a cat sat on a mat
    [4, 6, 7, 3, 4, 5],   # a dog ran on a mat
    [1, 7],                 # cat ran
    [6, 2],                 # dog sat
    [1, 2, 3, 5],           # cat sat on mat (implicit 'the')
    [6, 7, 3, 5],           # dog ran on mat
    [0, 1, 2, 3, 5],       # the cat sat on mat
    [0, 6, 7, 3, 5],       # the dog ran on mat
    [4, 1, 7],              # a cat ran
    [4, 6, 2],              # a dog sat
    [0, 1, 7, 3, 0, 5],   # the cat ran on the mat
    [0, 6, 2, 3, 0, 5],   # the dog sat on the mat
    [1, 2],                 # cat sat
    [6, 7],                 # dog ran
]

# --- Max sequence length ---
MAX_SEQ_LEN = max(len(s) for s in SENTENCES)

# --- Helper functions ---

def one_hot(token_id, size=VOCAB_SIZE):
    """One-hot encode a token ID."""
    v = np.zeros(size, dtype=np.float32)
    v[token_id] = 1.0
    return v


def token_to_bits(token_id):
    """Convert token ID (0-7) to 3-bit binary."""
    return [(token_id >> (2 - i)) & 1 for i in range(NUM_BITS)]


def bits_to_token(bits):
    """Convert 3-bit binary to token ID."""
    return (bits[0] << 2) | (bits[1] << 1) | bits[2]


def make_masked_examples(sentences, rng=None):
    """
    Generate (context, position, mask, target) training pairs.

    For each sentence, mask each position one at a time to create
    a training example where the model must predict the masked token
    given all other tokens.

    Returns list of (input_vector, target_token_id) pairs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    examples = []
    for sent in sentences:
        n = len(sent)
        for mask_pos in range(n):
            # Build input: for each position, one-hot token + mask flag + position encoding
            # input_dim per position = VOCAB_SIZE + 1 (mask flag) + 1 (position/MAX_SEQ_LEN)
            input_vec = build_input_vector(sent, mask_pos)
            target = sent[mask_pos]
            examples.append((input_vec, target))

    return examples


def build_input_vector(sentence, mask_pos):
    """
    Build the input vector for a single masked-position prediction.

    We encode ALL positions in the sentence into a fixed-size vector:
    - For each of MAX_SEQ_LEN positions:
      - 8 dims: one-hot token (zeroed if masked or empty)
      - 1 dim: mask flag (1 if this is the position to predict)
      - 1 dim: position / MAX_SEQ_LEN (normalized position)
    Total: MAX_SEQ_LEN * 10

    Plus 1 dim for sentence length / MAX_SEQ_LEN.
    """
    dims_per_pos = VOCAB_SIZE + 2  # 8 + 1 (mask) + 1 (pos)
    vec = np.zeros(MAX_SEQ_LEN * dims_per_pos + 1, dtype=np.float32)

    for i in range(len(sentence)):
        offset = i * dims_per_pos
        if i == mask_pos:
            # Mask flag = 1, token encoding = 0
            vec[offset + VOCAB_SIZE] = 1.0
        else:
            # One-hot encode the token
            vec[offset + sentence[i]] = 1.0
        # Normalized position
        vec[offset + VOCAB_SIZE + 1] = (i + 1) / MAX_SEQ_LEN

    # Sentence length
    vec[-1] = len(sentence) / MAX_SEQ_LEN

    return vec


INPUT_DIM = MAX_SEQ_LEN * (VOCAB_SIZE + 2) + 1


def decode_sentence(token_ids):
    """Convert token IDs to a readable sentence string."""
    return ' '.join(VOCAB[t] for t in token_ids)


if __name__ == '__main__':
    print(f"Vocabulary ({VOCAB_SIZE} tokens): {VOCAB}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")
    print(f"Input dimension: {INPUT_DIM}")
    print(f"Number of sentences: {len(SENTENCES)}")
    print()
    print("Training sentences:")
    for s in SENTENCES:
        print(f"  {s} -> {decode_sentence(s)}")
    print()
    print("Example masked training pair:")
    examples = make_masked_examples(SENTENCES[:2])
    inp, tgt = examples[0]
    print(f"  Input shape: {inp.shape}")
    print(f"  Target token: {tgt} ({VOCAB[tgt]})")
    print(f"  Total training examples: {len(make_masked_examples(SENTENCES))}")
