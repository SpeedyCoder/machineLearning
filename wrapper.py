import importlib

import reader
import practical4

batch_size = 1

vocab, E_talks, E_keywords, talks_dict, keywords_dict = reader.get_generation_data(1585, 250, 250)
batches_train = reader.make_batches_gen(talks_dict["train"], keywords_dict["train"], batch_size)

def start():
	importlib.reload(practical4)
	try:
		practical4.train(vocab, E_talks, E_keywords, talks_dict, 
						 keywords_dict, batches_train)
	except Exception as e:
		print(e)

