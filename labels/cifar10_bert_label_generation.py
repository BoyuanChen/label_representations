import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
from transformers import BertTokenizer, BertModel
from PIL import Image
from scipy.special import softmax

num_classes = 10

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

hidden_states = []
for p_label in labels:
    input_ids = torch.tensor(tokenizer.encode(p_label)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    hidden_states.append(last_hidden_states)

hidden_states_new = hidden_states.copy()
for i in range(len(hidden_states)):
    hidden_states_new[i] = hidden_states[i].squeeze(dim=0).view(2304).view(48, 48)

bert_labels = np.empty((len(hidden_states_new), 48, 48))
for i in range(10):
    bert_labels[i, :, :] = hidden_states_new[i].detach().numpy()

bert_labels[np.abs(bert_labels) > (np.mean(bert_labels) + 2 * np.std(bert_labels))] = 0

bert_labels = ((bert_labels - np.min(bert_labels)) / (np.max(bert_labels) - np.min(bert_labels)) * 80.0)


# Get average normalized pairwise difference:
avg_dist = 0
for i in range(10):
    for j in range(i + 1, 10):
        avg_dist += np.linalg.norm(bert_labels[i] - bert_labels[j], ord=1) / (
            (np.max(bert_labels) - np.min(bert_labels)) * bert_labels.shape[1] * bert_labels.shape[2]
        )


print("min: ", np.min(bert_labels))
print("max: ", np.max(bert_labels))
print("std: ", np.std(bert_labels))
print("average dist: ", avg_dist)
print(bert_labels.shape)

np.save("cifar10_bert.npy", bert_labels.astype(np.float32))

entropy_dict = {}
entropy_arr = []
for i in range(10):
    value = shannon_entropy(hidden_states_new[i].detach().numpy())
    entropy_dict[labels[i]] = value
    entropy_arr.append(value)

entropy_arr = np.asarray(entropy_arr)
print("mean entropy: ", np.mean(entropy_arr))
print("std entropy: ", np.std(entropy_arr))

for i in range(10):
    fig, ax = plt.subplots()
    ax.imshow(bert_labels[i], cmap="magma")
    fig.patch.set_visible(False)
    ax.axis("off")
    fig.savefig(f"./images/{labels[i]}.png")
    plt.close("all")