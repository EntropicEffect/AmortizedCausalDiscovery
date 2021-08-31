import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as tdist
from collections import defaultdict


def my_softmax(input, axis=1):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


def edge_accuracy(preds, target, binary=True):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    _, preds = preds.max(-1)
    if binary:
        preds = (preds >= 1).long()
    correct = preds.float().data.eq(target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))


def get_observed_relations_idx(num_atoms):
    length = (num_atoms ** 2) - num_atoms * 2
    remove_idx = np.arange(length)[:: num_atoms - 1][1:] - 1
    idx = np.delete(np.linspace(0, length - 1, length), remove_idx)
    return idx


def mse_per_sample(output, target):
    mse_per_sample = F.mse_loss(output, target, reduction="none")
    mse_per_sample = torch.mean(
        mse_per_sample, dim=(1, 2, 3)).cpu().data.numpy()
    return mse_per_sample


def edge_accuracy_per_sample(preds, target):
    _, preds = preds.max(-1)
    acc = torch.sum(torch.eq(preds, target), dim=1, dtype=torch.float64,) / preds.size(
        1
    )
    return acc.cpu().data.numpy()


def edge_accuracy_observed(preds, target, num_atoms=5):
    idx = get_observed_relations_idx(num_atoms)
    _, preds = preds.max(-1)
    correct = preds[:, idx].eq(target[:, idx]).cpu().sum()
    return np.float(correct) / (target.size(0) * len(idx))


def softplus(x):
    return torch.log(1.0 + torch.exp(x))


def distribute_over_GPUs(args, model, num_GPU=None):
    ## distribute over GPUs
    if args.device.type != "cpu":
        if num_GPU is None:
            model = torch.nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            args.batch_size_multiGPU = args.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = torch.nn.DataParallel(
                model, device_ids=list(range(num_GPU)))
            args.batch_size_multiGPU = args.batch_size * num_GPU
    else:
        model = torch.nn.DataParallel(model)
        args.batch_size_multiGPU = args.batch_size

    model = model.to(args.device)

    return model, num_GPU


def create_rel_matrix(args, num_atoms):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    if args.unobserved > 0 and args.model_unobserved == 1:
        num_atoms -= args.unobserved

    # Generate off-diagonal interaction graph
    rel_matrix = torch.Tensor(np.random.normal(size=(num_atoms, num_atoms)))

    if args.cuda:
        rel_matrix = rel_matrix.cuda()

    return rel_matrix


def append_losses(losses_list, losses):
    for loss, value in losses.items():
        if type(value) == float:
            losses_list[loss].append(value)
        elif type(value) == defaultdict:
            if losses_list[loss] == []:
                losses_list[loss] = defaultdict(list)
            for idx, elem in value.items():
                losses_list[loss][idx].append(elem)
        else:
            losses_list[loss].append(value.item())
    return losses_list


def average_listdict(listdict, num_atoms):
    average_list = [None] * num_atoms
    for k, v in listdict.items():
        average_list[k] = sum(v) / len(v)
    return average_list


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def sample_uniform_from_latents(latent_means, latent_width):
    latent_dist = tdist.uniform.Uniform(
        latent_means - latent_width, latent_means + latent_width
    )
    latents = latent_dist.rsample()
    return latents


def get_correlation(a, b):
    numerator = torch.sum((a - a.mean()) * (b - b.mean()))
    denominator = torch.sqrt(torch.sum((a - a.mean()) ** 2)) * torch.sqrt(
        torch.sum((b - b.mean()) ** 2)
    )
    return numerator / denominator
