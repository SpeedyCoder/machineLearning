import importlib
import traceback

import reader
import practical4


class Batches(object):
	def __init__(self, data, config):
		self.data = data
		self.config = config
		self.data.batches_train = []

	def set_batch_size(self, n):
		self.config.batch_size = n

	def get_batch_size(self):
		return self.config.batch_size

	def make_batches(self):
		self.data.batches_train = reader.make_batches_gen(
			self.data.talks_train, 
			self.data.keywords_train, 
			self.config.batch_size
		)
		self.data.batches_validate = reader.make_batches_gen(
			self.data.talks_validate, 
			self.data.keywords_validate, 
			self.config.batch_size
		)
		self.data.batches_test = reader.make_batches_gen(
			self.data.talks_test, 
			self.data.keywords_test, 
			self.config.batch_size
		)

	def get_train_batches(self):
		return self.data.batches_train


data = reader.get_generation_data(1585, 250, 250)
config = practical4.Config(data)
batches = Batches(data, config)
batches.make_batches()

def start():
	importlib.reload(practical4)
	try:
		model = practical4.Model(data, config)
		model.train()
	except Exception as e:
		traceback.print_exc()
		print(e)

