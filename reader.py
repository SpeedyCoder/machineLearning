import numpy as np
import os

import json

import re
from collections import Counter
from nltk.corpus import stopwords

from time import time

english = stopwords.words('english')


class Data(object):
    def __init__(self):
        self.vocab = []
        self.keys_vocab = []
        self.index_map = {}
        self.keys_index_map = {}
        self.E_talks = []
        self.E_keywords = []

        self.talks_train = []
        self.talks_validate = []
        self.talks_test = []

        self.keywords_train = []
        self.keywords_validate = []
        self.keywords_test = []


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


def _process_keywords(keywords):
    return keywords.replace(' ', '').lower().split(',')


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

    # for stopword in english:
    #     print(stopword)
    #     talk_tokens = filter(lambda x: x != stopword, talk_tokens)

    talk_tokens = [word for word in talk_tokens if word not in english]

    return (talk_tokens)


def _process_talk_gen(talk):
    # Remove text in parenthesis
    talk_noparens = re.sub(r'\([^)]*\)', '', talk)

    # Remove the names of the speakers
    sentences_strings = []
    for line in talk_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)

    talk_tokens = ["<START>"]
    for sent_str in sentences_strings:
        tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
        tokens.append("<EOS>")
        talk_tokens.extend(tokens)

    talk_tokens.append("<END>")

    return talk_tokens


def _to_vec_seq(word_to_id, talks, MAX_SIZE=None):
    def to_id(word):
        if word in word_to_id:
            return word_to_id[word]
        else:
            return 1
    print("Converting talks to vectors...")
    if MAX_SIZE is None:
        return [[to_id(word) for word in talk] for talk in talks]
    else:
        return[[to_id(word) for i, word in enumerate(talk) if i < MAX_SIZE] for talk in talks]


def _make_glove_embedding(words, keywords, embedding_size=200):
    print("Loading GloVe...")
    data = Data()
    data.vocab = ["<pad>", "<unk>", "<START>", "<EOS>", "<END>"]
    data.keys_vocab = ["<pad>", "<unk>"]
    data.index_map = {
        "<pad>": 0,
        "<unk>": 1,
        "<START>": 2,
        "<EOS>": 3,
        "<END>": 4
    }
    data.keys_index_map = {
        "<pad>": 0,
        "<unk>": 1
    }
    index = 5
    data.E_talks =( [np.zeros(embedding_size, dtype=np.float32) for _ in range(2)] +
            [2 * np.random.randn(embedding_size) for _ in range(3)])
    with open('glove.6B.200d.txt', encoding='utf-8') as f:
        for line in f:
            vec = line.split()
            word = vec.pop(0)
            if word in words:
                vec = np.array([float(r) for r in vec], dtype=np.float32)
                data.E_talks.append(vec)
                data.vocab.append(word)
                data.index_map[word] = index
                index += 1
                

    data.E_keywords = [np.zeros(50, dtype=np.float32) for _ in range(2)]
    index = 2
    with open('glove.6B.50d.txt', encoding='utf-8') as f:
        for line in f:
            vec = line.split()
            word = vec.pop(0)
            if word in keywords:
                vec = np.array([float(r) for r in vec], dtype=np.float32)
                data.E_keywords.append(vec)
                data.keys_vocab.append(word)
                data.keys_index_map[word] = index
                index += 1

    data.E_talks = np.array(data.E_talks, dtype=np.float32)
    data.E_keywords = np.array(data.E_keywords, dtype=np.float32)

    return data


def _make_random_embeddings(words, keywords, embedding_size=200):
    print("Making Random Embeddings...")
    data = Data()
    data.vocab = ["<pad>", "<unk>", "<START>", "<EOS>", "<END>"]
    data.keys_vocab = ["<pad>", "<unk>"]
    data.index_map = {
        "<pad>": 0,
        "<unk>": 1,
        "<START>": 2,
        "<EOS>": 3,
        "<END>": 4
    }
    data.keys_index_map = {
        "<pad>": 0,
        "<unk>": 1
    }
    index = 5
    data.E_talks =( [np.zeros(embedding_size, dtype=np.float32) for _ in range(2)] +
            [2 * np.random.randn(embedding_size) for _ in range(3)])
    
    for word in words:
        if word not in data.vocab:
            vec = 2 * np.random.randn(embedding_size)
            data.index_map[word] = index
            data.vocab.append(word)
            index += 1
            data.E_talks.append(vec)

    data.E_keywords = [np.zeros(embedding_size, dtype=np.float32) for _ in range(2)]
    index = 2
    for word in keywords:
        if word in data.index_map:
            vec = data.E_talks[data.index_map[word]]
        else:
            vec = 2 * np.random.randn(embedding_size)

        data.E_keywords.append(vec)
        data.keys_vocab.append(word)
        data.keys_index_map[word] = index
        index += 1


    data.E_talks = np.array(data.E_talks, dtype=np.float32)
    data.E_keywords = np.array(data.E_keywords, dtype=np.float32)

    return data



def _make_random_embedding(talks, embedding_size=20):
    index_map = {}
    index = 2
    # 0 is for padding and 1 unknown word 
    mat = [np.zeros(embedding_size, dtype=np.float32)]
    for talk in talks:
        for word in talk:
            if word not in index_map:
                vec = 2 * np.random.randn(embedding_size)
                mat.append(vec)
                index_map[word] = index
                index += 1

    return index_map, np.array(mat, dtype=np.float32)


def _pad(talk, length):
    return talk + ["<pad>" for _ in range(length-len(talk))]


def get_generation_data(n_train, n_validate, n_test, MAX_SIZE=None, voc_size=40000, keys_voc_size=330):
    start = time()
    if os.path.isfile('talks_gen.json'):
        print("Loading the data...")
        
        with open('talks_gen.json', 'r') as f:
            talks = json.load(f)

        with open('keywords_gen.json', 'r') as f:
            keywords = json.load(f)

    else:
        print("Processing the data...")
         # Download the dataset if it's not already there: this may take a minute as it is 75MB
        if not os.path.isfile('ted_en-20160408.zip'):
            import urllib.request
            print("Downloading the data...")
            urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

        
        import zipfile
        import lxml.etree
        # For now, we're only interested in the subtitle text, so let's extract that from the XML:
        with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
            doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

        talks = doc.xpath('//content/text()')
        keywords = doc.xpath('//head/keywords/text()')
        del doc

        keywords = list(map(_process_keywords, keywords))
        talks = list(map(_process_talk_gen, talks))
        print(list(map(len, talks[:20])))

        res = sorted(zip(talks, keywords), key=lambda x: len(x[0]))
        res = [(talk, keys) for talk, keys in res if len(talk) > 100]
        print("Talks:", len(res))
        keywords = [keys for _, keys in res]
        talks = [talk for talk, _ in res]
        del res
        print(list(map(len, talks[:20])))

        print(max(map(len, talks))) # => 7020

        # Save talks
        with open('talks_gen.json', 'w') as f:
            json.dump(talks, f)

        # Save keywords
        with open('keywords_gen.json', 'w') as f:
            json.dump(keywords, f)

    words = set(flatten(talks[:n_train]))
    keys = set(flatten(keywords[:n_train]))
    # all_words = flatten(talks[:n_train])
    # counter = Counter(all_words)
    # words = [word for word, _ in counter.most_common(voc_size)]
    # del all_words, counter
    # print(len(words))
    # all_keys = flatten(keywords[:n_train])
    # counter = Counter(all_keys)
    # keys = [key for key, _ in counter.most_common(keys_voc_size)]
    # del all_keys, counter
    # print(len(keys))

    data = _make_glove_embedding(words, keys)

    keywords = _to_vec_seq(data.keys_index_map, keywords)
    talks = _to_vec_seq(data.index_map, talks, MAX_SIZE=MAX_SIZE)

    data.talks_train = talks[:n_train]
    data.talks_validate = talks[n_train: n_train+n_validate]
    data.talks_test = talks[n_train+n_validate: n_train+n_validate+n_test]
    
    data.keywords_train = keywords[:n_train]
    data.keywords_validate = keywords[n_train: n_train+n_validate]
    data.keywords_test = keywords[n_train+n_validate: n_train+n_validate+n_test]

    end = time()
    print(end-start, "seconds")
    
    return data



def get_raw_data(n_train, n_validate, n_test, MAX_SIZE=None):
    if os.path.isfile('talks.json'):
        print("Loading the data...")
        start = time()
        with open('talks.json', 'r') as f:
            talks = json.load(f)
        
        keywords = np.load('keywords.npy')

        end = time()
        print(end-start, "seconds")

    else:
        print("Processing the data...")
         # Download the dataset if it's not already there: this may take a minute as it is 75MB
        if not os.path.isfile('ted_en-20160408.zip'):
            import urllib.request
            print("Downloading the data...")
            urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

        
        import zipfile
        import lxml.etree
        # For now, we're only interested in the subtitle text, so let's extract that from the XML:
        with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
            doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

        talks = doc.xpath('//content/text()')
        keywords = doc.xpath('//head/keywords/text()')
        del doc

        # Process keywords
        keywords = list(map(_make_label, keywords))
        # ooo 62% of training and 42.4% of validation
        res = []
        deleted = 0
        for i, talk in enumerate(talks):
            curr = _process_talk(talk)
            if len(curr) < MAX_SIZE:
                if len(curr) == 0:
                    keywords.pop(i)
                    deleted += 1
                else:
                    curr = _pad(curr, MAX_SIZE)
                    res.append(curr)
            else:
                res.append(curr)

            if i%100 == 0:
                print(i, "talks done")

        print("Deleted:", deleted)
        keywords = np.array(keywords)
        talks = res
        # print(max(map(len, talks))) => 2941

        # Save talks
        with open('talks.json', 'w') as f:
            json.dump(talks, f)

        # Save keywords
        np.save('keywords', keywords)

    index_map, vocab, E = _make_glove_embedding(talks[:n_train])
    talks = _to_vec_seq(index_map, talks, MAX_SIZE=MAX_SIZE)

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


def make_batches(talks, keywords, batch_size, equal_len=True, equal_size=True):
    batches = []
    for talks_batch, keywords_batch in _to_chunks(talks, keywords, batch_size):
        n_steps = max(map(len, talks_batch))
        # Make talks equally long
        if equal_len:
            talks_batch = [talk + [0 for _ in range(n_steps - len(talk))] for talk in talks_batch]
        batches.append((talks_batch, keywords_batch))

    if equal_size:
        if len(batches[-1][0]) < batch_size:
            batches.pop(-1)

    return batches

def make_batches_gen(talks, keywords, batch_size):
    batches = []
    for talks_batch, keywords_batch in _to_chunks(talks, keywords, batch_size):
        n_keywords = max(map(len, keywords_batch))
        keywords_batch = [keys + [0 for _ in range(n_keywords-len(keys))] 
                          for keys in keywords_batch]
        batch = {
            "inputs": [],
            "targets": [],
            "keywords": keywords_batch,
            "mask": [],
            "seq_lengths": list(map(lambda x: len(x)-1, talks_batch)),
            "max_len": 0
        }
        batch["max_len"] = max(batch["seq_lengths"])

        for talk in talks_batch:
            padding = [0 for _ in range(batch["max_len"]-len(talk)+1)]
            batch["mask"].append(
                [1 for _ in range(len(talk)-1)] + padding )
            batch["inputs"].append(
                talk[:len(talk)-1] + padding)
            batch["targets"].append(
                talk[1:]+ padding)

        batch["mask"] = np.array(batch["mask"], dtype=np.float32)
        batch["loss_weights"] = [np.ones(len(talks_batch)*batch["max_len"], dtype=np.float32)]
        batches.append(batch)

    return batches

def make_array(talks, keywords):
    batch = make_batches(talks, keywords, len(talks))
    return batch[0]





