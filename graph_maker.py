import matplotlib.pyplot as plt

def make_learning_curve(file_name, x, y1, y2, label1, label2):
	plt.plot(x, y1, marker='o', color='r', label=label1)
	plt.plot(x, y2, marker='x', color='g', label=label2)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend(loc=4)
	plt.title('Learning curve')

	plt.savefig(file_name)
	plt.clf()

