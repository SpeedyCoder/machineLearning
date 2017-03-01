import numpy as np

class Balancer(object):
	def __init__(self, X, Y):
		print("Min in X", min(map(len, X)))
		self.n_classes = Y.shape[1]
		classes = {}
		indexes = []
		lengths = []
		skip_class = []
		for c in range(self.n_classes):
			cond = Y.T[c] == 1
			classes[c] = {
				"x": [],
				"y": Y[cond]
			}
			for i, is_in in enumerate(cond):
				if is_in:
					classes[c]["x"].append(X[i])

			indexes.append(0)
			lengths.append(len(classes[c]["y"]))
			if len(classes[c]["y"]) == 0:
				skip_class.append(c)

		# for i, y in enumerate(Y):
		# 	print(i)
		# 	c = y.argmax()
		# 	classes[c]["x"].append(X[i].tolist())
		# 	classes[c]["y"].append(y.tolist())
		# 	lengths[c] += 1

		# for c in range(self.n_classes):
		# 	classes[c]["x"] = np.array(classes[c]["x"], dtype=np.float32)
		# 	classes[c]["y"] = np.array(classes[c]["y"], dtype=np.float32)

		print(lengths)
		self.classes = classes
		self.indexes = indexes
		self.skip_class = skip_class

	def next_batch(self, batch_size):
		classes = self.n_classes - len(self.skip_class)
		size = int(batch_size/classes)
		counts = [size for _ in range(self.n_classes)]

		for c in self.skip_class:
			counts[c] = 0

		left = batch_size - size*classes
		i = 0
		while left != 0:
			if i not in self.skip_class:
				counts[i] += 1
				left -= 1

			i += 1

		batch_x = None
		batch_y = None
		for c in range(self.n_classes):
			index = self.indexes[c]
			x = None
			y = None
			count = counts[c]
			while (count > 0):
				index = self.indexes[c]
				if x is None:
					x = self.classes[c]["x"][index: index + count]
					y = self.classes[c]["y"][index: index + count]
				else:
					x.extend(self.classes[c]["x"][index: index + count])
					y = np.concatenate((y, self.classes[c]["y"][index: index + count]))

				if index + count >= len(self.classes[c]["y"]):
					self.indexes[c] = 0
					count -= len(self.classes[c]["y"]) - index
				else:
					self.indexes[c] += count
					count = 0

			if x is not None:
				if batch_x is None:
					batch_x = x
					batch_y = y
				else:
					batch_x.extend(x)
					batch_y = np.concatenate((batch_y, y))

		# batch_x = np.array(batch_x, dtype=np.float32)
		# batch_y = np.array(batch_y, dtype=np.float32)

		return batch_x, batch_y





