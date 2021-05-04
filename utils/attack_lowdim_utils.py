import torch
import torch.nn as nn


# Checks if the nearest neighbor of the output is the target
def is_nn_target(output, target, mels):
    mse = nn.MSELoss(reduction="none")
    num_classes = mels.shape[0]
    mse_dists = mse(output.repeat(num_classes, 1), mels).mean(-1)  # low-dim label is 1D
    output_NN = mels[torch.argmin(mse_dists)]
    return (output_NN - target).abs().sum() < 1e-5


# FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain image range
    perturbed_image = torch.clamp(
        perturbed_image, image.min().item(), image.max().item()
    )
    # Return the perturbed image
    return perturbed_image


# FGSM (untargeted)
def test_fgsm_untargeted(model, device, test_loader, epsilon, mels):
    # Accuracy counter
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        # Calculate the loss
        init_loss = nn.functional.smooth_l1_loss(output, target)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if not is_nn_target(output, target, mels):
            continue
        elif epsilon == 0:
            correct += 1
            if len(adv_examples) < 30:
                # Save some examples for visualization later
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_loss.item(), init_loss.item(), adv_ex))
            continue

        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        init_loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        if is_nn_target(output, target, mels):
            correct += 1
        elif len(adv_examples) < 30:
            # Save some adv examples for visualization later
            final_loss = nn.functional.smooth_l1_loss(output, target)
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_loss.item(), final_loss.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


# FGSM (targeted)
def test_fgsm_targeted(model, num_classes, device, test_loader, epsilon, mels):
    # Accuracy counter
    correct = 0
    adv_examples = []
    gen = torch.manual_seed(444)

    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        adv_target_idx = torch.randint(0, num_classes, (1,), generator=gen).item()
        mel = mels[adv_target_idx : adv_target_idx + 1]
        while (mel - target).abs().sum() < 1e-5:
            adv_target_idx = torch.randint(0, num_classes, (1,), generator=gen).item()
            mel = mels[adv_target_idx : adv_target_idx + 1]
        adv_target = mel.clone().to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        # Calculate the loss
        init_loss = nn.functional.smooth_l1_loss(output, target)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if not is_nn_target(output, target, mels):
            continue
        elif epsilon == 0:
            correct += 1
            if len(adv_examples) < 30:
                # Save some examples for visualization later
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_loss.item(), init_loss.item(), adv_ex))
            continue

        adv_loss = nn.functional.smooth_l1_loss(output, adv_target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        adv_loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, -epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        if is_nn_target(output, target, mels):
            correct += 1
        elif len(adv_examples) < 30:
            # Save some adv examples for visualization later
            final_loss = nn.functional.smooth_l1_loss(output, target)
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_loss.item(), final_loss.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


# Basic iterative attack
def iterative_attack(image, last_perturbed_image, epsilon, alpha, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = last_perturbed_image + alpha * sign_data_grad
    # Adding clipping to maintain [-epsilon,epsilon] range for accumulated gradients
    total_grad = perturbed_image - image
    total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    perturbed_image = image + total_grad
    # perturbed_image = torch.clamp(perturbed_image, image.min().item(), image.max().item())
    # Return the perturbed image
    return perturbed_image.clone()


# iterative (untargeted)
def test_iterative_untargeted(
    model, device, test_loader, mels, epsilon, alpha, num_steps
):
    # Accuracy counter
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        # Calculate the loss
        init_loss = nn.functional.smooth_l1_loss(output, target)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if not is_nn_target(output, target, mels):
            continue
        elif epsilon == 0:
            correct += 1
            if len(adv_examples) < 30:
                # Save some examples for visualization later
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_loss.item(), init_loss.item(), adv_ex))
            continue

        orig_data = data.clone()
        perturbed_data = data
        for i in range(num_steps):
            # Calculate the loss
            loss = nn.functional.smooth_l1_loss(output, target)
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = perturbed_data.grad.data
            # Call Iterative Attack
            perturbed_data.data = iterative_attack(
                orig_data, perturbed_data, epsilon, alpha, data_grad
            )
            # Re-classify the perturbed image
            output = model(perturbed_data)

        # Check for success
        if is_nn_target(output, target, mels):
            correct += 1
        elif len(adv_examples) < 30:
            # Save some adv examples for visualization later
            final_loss = nn.functional.smooth_l1_loss(output, target)
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_loss.item(), final_loss.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


# iterative (targeted)
def test_iterative_targeted(
    model, num_classes, device, test_loader, mels, epsilon, alpha, num_steps
):
    # Accuracy counter
    correct = 0
    adv_examples = []
    gen = torch.manual_seed(444)

    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        adv_target_idx = torch.randint(0, num_classes, (1,), generator=gen).item()
        mel = mels[adv_target_idx : adv_target_idx + 1]
        while (mel - target).abs().sum() < 1e-5:
            adv_target_idx = torch.randint(0, num_classes, (1,), generator=gen).item()
            mel = mels[adv_target_idx : adv_target_idx + 1]
        adv_target = mel.clone().to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        # Get the index of the max log-probability
        init_loss = nn.functional.smooth_l1_loss(output, target)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if not is_nn_target(output, target, mels):
            continue
        elif epsilon == 0:
            correct += 1
            if len(adv_examples) < 30:
                # Save some examples for visualization later
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_loss.item(), init_loss.item(), adv_ex))
            continue

        orig_data = data.clone()
        perturbed_data = data
        for i in range(num_steps):
            # Calculate the loss
            loss = nn.functional.smooth_l1_loss(output, adv_target)
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = perturbed_data.grad.data
            # Call Iterative Attack
            perturbed_data.data = iterative_attack(
                orig_data, perturbed_data, epsilon, -alpha, data_grad
            )
            # Re-classify the perturbed image
            output = model(perturbed_data)

        # Check for success
        if is_nn_target(output, target, mels):
            correct += 1
        elif len(adv_examples) < 30:
            # Save some adv examples for visualization later
            final_loss = nn.functional.smooth_l1_loss(output, target)
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_loss.item(), final_loss.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
