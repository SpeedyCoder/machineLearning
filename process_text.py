import numpy as np
import os
import re

import urllib.request
import zipfile
import lxml.etree

from time import time

labels = ["ooo", "ooD", "oEo", "oED", "Too", "ToD", "TEo", "TED"]

def make_label(keywords):
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


def make_multi_label(keywords):
    keywords = keywords.replace(' ', '').lower().split(',')

    x = np.zeros(3, dtype=np.float32)
    if "technology" in keywords:
        x[0] = 1
    if "entertainment" in keywords:
        x[1] = 1
    if "design" in keywords:
        x[2] = 1

    return x


def flatten(l):
    return [elem for sub_l in l for elem in sub_l]


def process_talk(talk):
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


def make_glove_embedding(talks):
    print("Loading GloVe...")
    index_map = {}
    index = 0
    mat = []
    words = set(flatten(talks))
    with open('glove/glove.6B.50d.txt') as f:
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


def make_random_embedding(talks):
    index_map = {}
    index = 2
    # 0 is for no word and 1 for unknown word 
    mat = [np.zeros(50, dtype=np.float32), np.zeros(50, dtype=np.float32)]
    for talk in talks:
        for word in talk:
            if word not in index_map:
                vec = 2 * np.random.randn(50)
                mat.append(vec)
                index_map[word] = index
                index += 1

    return index_map, np.array(mat, dtype=np.float32)



def load_glove_vocab():
    vocab = {}
    with open('glove/glove.6B.50d.txt') as f:
        for line in f:
            vec = line.split()
            word = vec.pop(0)
            vec = np.array([float(r) for r in vec])
            vocab[word] = vec

    return vocab


def talks_to_vecs_simple(vocab, talks):
    print("Converting talks to vectors...")
    talks_vec = []
    for talk in talks:
        x = np.zeros(50)
        for word in talk:
            if word in vocab:
                x = np.add(x, vocab[word])
                # Representing unknown words with 0 vector

        if len(talk) > 0:
            x = x/len(talk)
        
        talks_vec.append(x)

    return np.array(talks_vec)


def talks_to_vec(index_map, vec_size, talks):
    print("Converting talks to vectors...")
    talks_vec = []
    for i, talk in enumerate(talks):
        if i%50 == 0:
            print("%s talks done"%i)
        x = np.zeros(vec_size, dtype=np.float32)
        for word in talk:
            if word in index_map:
                x[index_map[word]] += 1
            else:
                x[vec_size-1] += 1

        if len(talk) != 0:
            x = x/len(talk)

        talks_vec.append(x)

    return np.array(talks_vec)


def talks_to_vec_seq(index_map, talks):
    print("Converting talks to vectors...")
    vec_size = max(map(len, talks))
    print("Vector size:", vec_size)
    talks_vec = []
    for i, talk in enumerate(talks):
        if i%50 == 0:
            print("%s talks done"%i)
        x = np.zeros(vec_size, dtype=np.int32)
        # x = np.full(vec_size, -1, dtype=np.float32)
        for j, word in enumerate(talk):
            if word in index_map:
                x[j] = index_map[word]
            else:
                x[j] = 1

        talks_vec.append(x)

    return np.array(talks_vec)

def talks_to_vec_seq(index_map, E, talks):
    print("Converting talks to vectors...")
    vec_size = max(map(len, talks))
    print("Vector size:", vec_size)
    talks_vec = []
    for i, talk in enumerate(talks):
        if i%50 == 0:
            print("%s talks done"%i)
        x = np.zeros([vec_size, 50], dtype=np.float32)

        for j, word in enumerate(talk):
            if word in index_map:
                x[j] = E[index_map[word]]

        talks_vec.append(x)

    return np.array(talks_vec)


def prepare_data():
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
    keywords = np.array(list(map(make_label, keywords)))
    # print(keywords)
    talks = list(map(process_talk, talks))
    
    # Simple embedding
    # vocab = load_glove_vocab()
    # talks = talks_to_vecs_simple(vocab, talks)

    # Glove embedding
    # index_map, E = make_glove_embedding(talks[:1585])
    # talks = talks_to_vec(index_map, len(index_map)+1, talks)

    # Random embedding
    # index_map, E = make_random_embedding(talks[:1585])
    # talks = talks_to_vec(index_map, len(index_map)+1, talks)

    # Random embedding sequential
    index_map, E = make_random_embedding(talks[:1585])
    talks = talks_to_vec_seq(index_map, E, talks)


    print("Writing files...")
    np.save("keywords", keywords)
    # Simple embedding
    # np.save("talks_simple", talks)

    # Glove embedding
    # np.save("embedding_glove", E)
    # np.save("talks_glove", talks)

    # Random embedding
    # np.save("embedding_random", E)
    # np.save("talks_random", talks)

    # Random embedding sequential
    np.save("embedding", E)
    np.save("talks", talks)


start = time()
prepare_data()
end = time()
print("Done in %s s." % (end-start))

