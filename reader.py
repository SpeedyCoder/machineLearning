import numpy as np
import os
import re

import urllib.request
import zipfile
import lxml.etree


def flatten(l):
    return [elem for sub_l in l for elem in sub_l]


labels = ["ooo", "ooD", "oEo", "oED", "Too", "ToD", "TEo", "TED"]

def _make_label(keywords):
    keywords = keywords.replace(' ', '').lower().split(',')

    index = 0
    if "technology" in keywords:
        index += 4
    if "entertainment" in keywords:
        index += 2
    if "design" in keywords:
        index += 1

    x = np.zeros(8, dtype=np.float32)
    x[index] = 1.

    return x


def _process_talk(talk):
    # Remove text in parenthesis
    talk_noparens = re.sub(r'\([^)]*\)', '', talk)

    # Remove the names of the speakers
    sentences_strings = []
    for line in talk_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)

    talk_tokens = []
    for sent_str in sentences_strings:
        tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
        talk_tokens.extend(tokens)

    return talk_tokens


def _talks_to_vec_seq(word_to_id, talks, MAX_SIZE=None):
    print("Converting talks to vectors...")
    def to_id(word):
        if word in word_to_id:
            return word_to_id[word]
        else:
            return 1

    if MAX_SIZE is None:
        return [[to_id(word) for word in talk] for talk in talks]
    else:
        return[[to_id(word) for i, word in enumerate(talk) if i < MAX_SIZE] for talk in talks]


def _make_glove_embedding(talks):
    print("Loading GloVe...")
    index_map = {}
    index = 0
    mat = [np.zeros(50, dtype=np.float32), np.zeros(50, dtype=np.float32)]
    words = set(flatten(talks))
    with open('glove.6B.50d.txt', encoding='utf-8') as f:
        for line in f:
            vec = line.split()
            word = vec.pop(0)
            if word in words:
                vec = np.array([float(r) for r in vec], dtype=np.float32)
                index_map[word] = index
                index += 1
                mat.append(vec)

    # Unknown words
    mat.append(np.zeros(50, dtype=np.float32))

    return index_map, np.array(mat)


def get_raw_data(n_train, n_validate, n_test, MAX_SIZE=None):
    # Download the dataset if it's not already there: this may take a minute as it is 75MB
    if not os.path.isfile('ted_en-20160408.zip'):
        print("Downloading the data...")
        urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

    print("Processing the data...")
    # For now, we're only interested in the subtitle text, so let's extract that from the XML:
    with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
        doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

    talks = doc.xpath('//content/text()')
    keywords = doc.xpath('//head/keywords/text()')
    del doc

    # Process keywords
    keywords = np.array(list(map(_make_label, keywords)))
    talks = list(map(_process_talk, talks))

    index_map, E = _make_glove_embedding(talks[:n_train])
    talks = _talks_to_vec_seq(index_map, talks, MAX_SIZE=MAX_SIZE)

    talks_dict = {
        "train": talks[:n_train],
        "validate": talks[n_train: n_train+n_validate],
        "test": talks[n_train+n_validate: n_train+n_validate+n_test]
    }

    keywords_dict = {
        "train": keywords[:n_train],
        "validate": keywords[n_train: n_train+n_validate],
        "test": keywords[n_train+n_validate: n_train+n_validate+n_test]
    }
    
    return E, talks_dict, keywords_dict



def _to_chunks(l1, l2, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l1), n):
        yield l1[i:i + n], l2[i: i + n]


def make_batches(talks, keywords, batch_size):
    batches = []
    for talks_batch, keywords_batch in _to_chunks(talks, keywords, batch_size):
        n_steps = max(map(len, talks_batch))
        # Make talks equally long
        talks_batch = [talk + [0 for _ in range(n_steps - len(talk))] for talk in talks_batch]
        batches.append((talks_batch, keywords_batch))

    return batches

def make_array(talks, keywords):
    batch = make_batches(talks, keywords, len(talks))
    return batch[0]

# import numpy as np

# print("Loading data...")
# X = np.load("talks.npy")
# print("Shape of X:", X.shape)
# Y = np.load("keywords.npy")
# print("Shape of y:", Y.shape)
# E = np.load("embedding.npy")
# print("Shape of E:", Y.shape)

# Y_train = Y[:1585]
# X_train = X[:1585]

# Y_validation = Y[1585:1835]
# X_validation = X[1585:1835]

# Y_test = Y[1835:]
# X_test = X[1835:]





