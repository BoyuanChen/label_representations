import csv
import os
from pathlib import Path
import torch
import torch.nn as nn


# Directory setup for training
def train_directory_setup(
    label,
    model_name,
    dataset,
    seq_seed,
    data_level,
    base_dir,
):

    base_dir = Path(base_dir)
    if data_level < 100:
        base_folder = base_dir / "less/{}/level{}/seed{}/{}/model_{}".format(
            dataset, data_level, seq_seed, model_name, label
        )
        log_folder = base_dir / "less/log"
        log_path = os.path.join(
            log_folder, "{}_less_data_{}_log.csv".format(dataset, label)
        )
    else:
        base_folder = base_dir / "{}/seed{}/{}/model_{}".format(
            dataset, seq_seed, model_name, label
        )
        log_folder = base_dir / "log"
        log_path = os.path.join(log_folder, "{}_{}_log.csv".format(dataset, label))
    snapshots_folder = os.path.join(base_folder, "snapshots")

    # Makes folders
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(snapshots_folder, exist_ok=True)

    best_model_file = "{}_seed{}_{}_best_model.pth".format(label, seq_seed, model_name)
    checkpoint_file = "{}_seed{}_{}_checkpoint.pth".format(label, seq_seed, model_name)

    best_model_path = os.path.join(base_folder, best_model_file)
    checkpoint_path = os.path.join(base_folder, checkpoint_file)
    print("Best model location: {}.".format(best_model_path))
    print("Checkpoint location: {}.".format(checkpoint_path))
    print("Log location: {}.".format(log_path))
    print("Snapshots location: {}".format(snapshots_folder))
    return best_model_path, checkpoint_path, log_path, snapshots_folder



# Logs training results
def train_log_results(
    log_path, model_name, data_level, seq_seed, test_loss, test_acc, nn_acc=None
):
    less_data = data_level < 100
    if not os.path.exists(log_path):
        with open(log_path, "w") as log_file:
            log_writer = csv.writer(log_file, delimiter=",")
            if less_data:
                header = ["Model", "Data Level", "Seed", "Test Loss", "Test Acc"]
            else:
                header = ["Model", "Seed", "Test Loss", "Test Acc"]
            if nn_acc is not None:
                header.append("NN Acc")
            log_writer.writerow(header)

    with open(log_path, "a") as log_file:
        log_writer = csv.writer(log_file, delimiter=",")
        if less_data:
            row = [model_name, data_level, seq_seed, test_loss, test_acc]
        else:
            row = [model_name, seq_seed, test_loss, test_acc]
        if nn_acc is not None:
            row.append(nn_acc)
        log_writer.writerow(row)


def count_correct_nn(outputs, targets, mels):
    mse = nn.MSELoss(reduction="none")
    batch_size = outputs.size(0)
    num_classes = mels.size(0)
    outputs_repeated = outputs.unsqueeze(1).repeat_interleave(num_classes, dim=1)
    mels_repeated = mels.unsqueeze(0).repeat_interleave(batch_size, dim=0)
    if len(outputs.shape) == 2:
        mse_dists = mse(outputs_repeated, mels_repeated).mean(-1)
        outputs_NN = mels[mse_dists.argmin(-1)]

        return ((outputs_NN - targets).abs().sum(-1) < 1e-5).sum().item()

    else:
        mse_dists = mse(outputs_repeated, mels_repeated).mean(-1).mean(-1)
        outputs_NN = mels[mse_dists.argmin(-1)]

        return ((outputs_NN - targets).abs().sum(-1).sum(-1) < 1e-5).sum().item()


def train(model, trainloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    batch_count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        batch_count += 1
    return train_loss / batch_count


def valid_highdim(model, validloader, criterion, device):
    model.eval()
    smoothl1 = nn.SmoothL1Loss(reduction="none")
    mels = torch.tensor(validloader.dataset.mels, device=device)
    correct_loss = 3.5
    correct = 0
    correct_nn = 0
    test_total = 0
    valid_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Count correct predictions using threshold
            smoothl1_losses = smoothl1(outputs, targets)
            for i in range(len(smoothl1_losses)):
                test_total += 1
                curr_loss = smoothl1_losses[i].mean().cpu().numpy()
                if curr_loss < correct_loss:
                    correct += 1
            # Count correct predictions using 1-NN
            correct_nn += count_correct_nn(outputs, targets, mels)

            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            batch_count += 1

    return valid_loss / batch_count, correct_nn / test_total


def valid_category(model, validloader, criterion, device):
    model.eval()
    correct = 0
    test_total = 0
    valid_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            batch_count += 1
    return valid_loss / batch_count, correct / test_total


def valid_lowdim(model, validloader, criterion, device):
    model.eval()
    mels = torch.tensor(validloader.dataset.mels, device=device)
    correct = 0
    test_total = 0
    valid_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_total += targets.size(0)
            # Count correct predictions using 1-NN
            correct += count_correct_nn(outputs, targets, mels)

            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            batch_count += 1
    return valid_loss / batch_count, correct / test_total


def test_highdim(model, testloader, criterion, device):
    model.eval()
    smoothl1 = nn.SmoothL1Loss(reduction="none")
    mels = torch.tensor(testloader.dataset.mels, device=device)
    correct_loss = 3.5
    correct = 0
    correct_nn = 0
    test_total = 0
    test_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Count correct predictions using threshold
            smoothl1_losses = smoothl1(outputs, targets)
            for i in range(len(smoothl1_losses)):
                test_total += 1
                curr_loss = smoothl1_losses[i].mean().cpu().numpy()
                if curr_loss < correct_loss:
                    correct += 1
            # Count correct predictions using 1-NN
            correct_nn += count_correct_nn(outputs, targets, mels)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            batch_count += 1

    return test_loss / batch_count, correct_nn / test_total


def test_category(model, testloader, criterion, device):
    model.eval()
    correct = 0
    test_total = 0
    test_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            batch_count += 1
    return test_loss / batch_count, correct / test_total


def test_lowdim(model, testloader, criterion, device):
    model.eval()
    mels = torch.tensor(testloader.dataset.mels, device=device)
    correct = 0
    test_total = 0
    test_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_total += targets.size(0)
            # Count correct predictions using 1-NN
            correct += count_correct_nn(outputs, targets, mels)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            batch_count += 1
    return test_loss / batch_count, correct / test_total
