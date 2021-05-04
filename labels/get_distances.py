import numpy as np


num_classes = 100
labels = [
    f"cifar{num_classes}_speech.npy",
    f"cifar{num_classes}_uniform.npy",
    f"cifar{num_classes}_random.npy",
    f"cifar{num_classes}_composite.npy",
    f"cifar{num_classes}_shuffle.npy",
    f"cifar{num_classes}_lowdim.npy",
    f"cifar{num_classes}_bert.npy",
    f"cifar{num_classes}_glove.npy",
]

for label_name in labels:
    label = np.load(label_name)
    # Normalize to 0, 1
    label = (label - np.min(label)) / (np.max(label) - np.min(label))
    print("----------------------")
    avg_dist_list = []
    l2_dist_list = []
    l1_dist_list = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            l2_dist_list.append(np.linalg.norm(label[i] - label[j], ord=1))
            l1_dist_list.append(np.linalg.norm(label[i] - label[j], ord=2))

            if len(label.shape) == 3:
                dist = np.sum(np.abs(label[i] - label[j])) / (
                    (np.max(label) - np.min(label)) * label.shape[1] * label.shape[2]
                )

            elif len(label.shape) == 2:
                dist = np.sum(np.abs(label[i] - label[j])) / (
                    (np.max(label) - np.min(label)) * label.shape[1]
                )

            avg_dist_list.append(dist)

    print(f"{label_name} min: {np.min(label)}")
    print(f"{label_name} max: {np.max(label)}")
    print(f"{label_name} std: {np.std(label)}")
    print(
        f"{label_name} average dist: {np.mean(avg_dist_list)}",
    )

    print(
        f"{label_name} L2 dist: {np.mean(l2_dist_list)}",
    )
    print(
        f"{label_name} L2 dist std: {np.std(l2_dist_list)}",
    )

    print(
        f"{label_name} L1 dist: {np.mean(l1_dist_list)}",
    )
    print(
        f"{label_name} L1 dist std: {np.std(l1_dist_list)}",
    )

    print(f"{label_name} shape: {label.shape}")
    print("----------------------")
