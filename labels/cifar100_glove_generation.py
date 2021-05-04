import numpy as np
from skimage.measure import shannon_entropy
from tqdm import tqdm

def read_label_string():
    label_path = "./cifar100_text.txt"
    f = open(label_path, "r")
    labels = f.read()
    labels = labels.split(",")
    labels = [i.rstrip().lstrip() for i in labels]
    # labels = [i.split(' ')[0] for i in labels]
    return labels


def get_vec_from_names(names):
    label_arr = []
    label_path = "./glove.6B.50d.txt"
    f = open(label_path, "r")
    labels = f.read()
    labels = labels.split("\n")
    # Now construct a dictionary
    word_dict = {}
    for i, row in enumerate(tqdm(labels)):
        row = row.split(' ')
        word_dict[row[0]] = np.asarray(row[1:]).astype(np.float32)

    # Now retrieve vectors from names
    for name in names:
        name = name.split(' ')[0].split('-')[0]
        label_arr.append(word_dict[name])

    return np.asarray(label_arr)


cifar100_names = read_label_string()
cifar100_labels = get_vec_from_names(cifar100_names)

np.save('./cifar100_glove.npy', cifar100_labels.astype(np.float32))


entropy_dict = {}
entropy_arr = []
for i in range(10):
    value = shannon_entropy(cifar100_labels[i])
    entropy_dict[cifar100_names[i]] = value
    entropy_arr.append(value)

entropy_arr = np.asarray(entropy_arr)
print('mean entropy: ', np.mean(entropy_arr))
print('std entropy: ', np.std(entropy_arr))