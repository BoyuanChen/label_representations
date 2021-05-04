import torch
import torch.nn as nn

# from torchtoolbox.nn import LabelSmoothingLoss

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
def test_fgsm_untargeted(model, device, test_loader, epsilon, mels=None):
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
        # Get the index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        elif epsilon == 0:
            correct += 1
            if len(adv_examples) < 30:
                # Save some examples for visualization later
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), init_pred.item(), adv_ex))
            continue

        # Calculate the loss
        loss = nn.functional.cross_entropy(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
        elif len(adv_examples) < 30:
            # Save some adv examples for visualization later
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

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
def test_fgsm_targeted(model, num_classes, device, test_loader, epsilon, mels=None):
    # Accuracy counter
    correct = 0
    adv_examples = []
    gen = torch.manual_seed(444)

    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        adv_target = torch.randint(0, num_classes, (1,), generator=gen)
        while adv_target.item() == target.item():
            adv_target = torch.randint(0, num_classes, (1,), generator=gen)
        adv_target = adv_target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        # Get the index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        elif epsilon == 0:
            correct += 1
            if len(adv_examples) < 30:
                # Save some examples for visualization later
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), init_pred.item(), adv_ex))
            continue

        adv_loss = nn.functional.cross_entropy(output, adv_target)
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
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
        elif len(adv_examples) < 30:
            # Save some adv examples for visualization later
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

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
def test_iterative_untargeted(model, device, test_loader, mels, epsilon, alpha, num_steps):
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
        # Get the index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        elif epsilon == 0:
            correct += 1
            if len(adv_examples) < 30:
                # Save some examples for visualization later
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), init_pred.item(), adv_ex))
            continue

        orig_data = data.clone()
        perturbed_data = data
        for i in range(num_steps):
            # Calculate the loss
            loss = nn.functional.cross_entropy(output, target)
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
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
        elif len(adv_examples) < 30:
            # Save some adv examples for visualization later
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

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
        adv_target = torch.randint(0, num_classes, (1,), generator=gen)
        while adv_target.item() == target.item():
            adv_target = torch.randint(0, num_classes, (1,), generator=gen)
        adv_target = adv_target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        # Get the index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        elif epsilon == 0:
            correct += 1
            if len(adv_examples) < 30:
                # Save some examples for visualization later
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), init_pred.item(), adv_ex))
            continue

        orig_data = data.clone()
        perturbed_data = data
        for i in range(num_steps):
            # Calculate the loss
            loss = nn.functional.cross_entropy(output, adv_target)
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
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
        elif len(adv_examples) < 30:
            # Save some adv examples for visualization later
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
