#%load_ext autoreload
#%autoreload 2
import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random

## Fake dataset:

class Sampler:
    def __init__(self, prob_table):

        total_prob = 0.0

        if type(prob_table) is dict:
            for key, value in prob_table.items():
                total_prob += value
        elif type(prob_table) is list:
            prob_table_gen = {}
            for key in prob_table:
                prob_table_gen[key] = 1.0 / (float(len(prob_table)))
            total_prob = 1.0
            prob_table = prob_table_gen
        else:
            raise ArgumentError("__init__ takes either a dict or a list as its first argument")

        if total_prob <= 0.0:
            raise ValueError("Probability is not strictly positive.")

        self._keys = []
        self._probs = []

        for key in prob_table:
            self._keys.append(key)
            self._probs.append(prob_table[key] / total_prob)

    def __call__(self):

        sample = random.random()

        seen_prob = 0.0

        for key, prob in zip(self._keys, self._probs):
            if (seen_prob + prob) >= sample:
                return key
            else:
                seen_prob += prob
        return key

samplers = {
    "punctuation": Sampler({".": 0.49, ",": 0.5, ";": 0.03, "?": 0.05, "!": 0.05}),
    "stop": Sampler({"the": 10, "from": 5, "a": 9, "they": 3, "he": 3, "it" : 2.5, "she": 2.7, "in": 4.5}),
    "noun": Sampler(["cat", "broom", "boat", "dog", "car", "wrangler", "mexico", "lantern", "book", "paper", "joke","calendar", "ship", "event"]),
    "verb": Sampler(["ran", "stole", "carried", "could", "would", "do", "can", "carry", "catapult", "jump", "duck"]),
    "adverb": Sampler(["rapidly", "calmly", "cooly", "in jest", "fantastically", "angrily", "dazily"])
    }

def generate_nonsense(word = ""):
    if word.endswith("."):
        return word
    else:
        if len(word) > 0:
            word += " "

        word += samplers["stop"]()
        word += " " + samplers["noun"]()

        if random.random() > 0.7:
            word += " " + samplers["adverb"]()
            if random.random() > 0.7:
                word += " " + samplers["adverb"]()

        word += " " + samplers["verb"]()

        if random.random() > 0.8:
            word += " " + samplers["noun"]()
            if random.random() > 0.9:
                word += "-" + samplers["noun"]()

        if len(word) > 500:
            word += "."
        else:
            word += " " + samplers["punctuation"]()

        return generate_nonsense(word)

def generate_dataset(total_size, ):
    sentences = []
    for i in range(total_size):
        sentences.append(generate_nonsense())
    return sentences

# generate dataset
lines = generate_dataset(5)

### Utilities:
class Vocab:
    __slots__ = ["word2index", "index2word", "unknown"]

    def __init__(self, index2word = None):
        self.word2index = {}
        self.index2word = []

        # add unknown word:
        self.add_words(["**UNKNOWN**"])
        self.unknown = 0

        if index2word is not None:
            self.add_words(index2word)

    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)

    def __call__(self, line):
        """
        Convert from numerical representation to words
        and vice-versa.
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = line.split(" ")
            indices = np.zeros(len(line), dtype=np.int32)

        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)

        return indices

    @property
    def size(self):
        return len(self.index2word)

    def __len__(self):
        return len(self.index2word)

vocab = Vocab()
for line in lines:
    vocab.add_words(line.split(" "))

def pad_into_matrix(rows, padding = 0):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = map(len, rows)
    width = max(lengths)
    height = len(rows)
    mat = np.empty([height, width], dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
        mat[i, 0:len(row)] = row
    return mat, list(lengths)

# transform into big numerical matrix of sentences:
numerical_lines = []
for line in lines:
    numerical_lines.append(vocab(line))
numerical_lines, numerical_lengths = pad_into_matrix(numerical_lines)

print lines
print numerical_lines
print numerical_lengths
