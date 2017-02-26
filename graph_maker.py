import matplotlib.pyplot as plt

def make_learning_curve(file_name, x, y1, y2, label1, label2):
    plt.plot(x, y1, color='r', label=label1)
    plt.plot(x, y2, color='g', label=label2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    plt.title('Learning curve')
    plt.savefig(file_name)
    plt.clf()

if __name__ == "__main__":
    import numpy as np

    accs_train = np.load("accs_train.npy")
    accs_validate = np.load("accs_validate.npy")

    make_learning_curve("graph.png", range(0, 100), accs_train, accs_validate, "training", "validation")
