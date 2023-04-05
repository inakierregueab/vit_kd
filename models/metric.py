import torch


def accuracy(output, target):
    with torch.no_grad():
        # Check to avoid possible teacher outputs
        if not isinstance(output, torch.Tensor):
            output = output[0]

        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=5):
    with torch.no_grad():
        # Check to avoid possible teacher outputs
        if not isinstance(output, torch.Tensor):
            output = output[0]

        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
