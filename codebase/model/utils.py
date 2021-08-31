import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
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
    rel_matrix = torch.Tensor(np.random.normal(seize=(num_atoms, num_atoms)))

    if args.cuda:
        rel_maxtrix = rel_matrix.cuda()

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


# Latent Temperature Experiment utils
def get_uniform_parameters_from_latents(latent_params):
    n_params = latent_params.shape[1]
    logit_means = latent_params[:, : n_params // 2]
    logit_widths = latent_params[:, n_params // 2:]
    means = sigmoid(logit_means)
    widths = sigmoid(logit_widths)
    mins, _ = torch.min(
        torch.cat([means, 1 - means], dim=1), dim=1, keepdim=True)
    widths = mins * widths
    return means, widths


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def sample_uniform_from_latents(latent_means, latent_width):
    latent_dist = tdist.uniform.Uniform(
        latent_means - latent_width, latent_means + latent_width
    )
    latents = latent_dist.rsample()
    return latents


def get_categorical_temperature_prior(mid, num_cats, to_torch=True, to_cuda=True):
    categories = [mid * (2.0 ** c)
                  for c in np.arange(num_cats) - (num_cats // 2)]
    if to_torch:
        categories = torch.Tensor(categories)
    if to_cuda:
        categories = categories.cuda()
    return categories


def kl_uniform(latent_width, prior_width):
    eps = 1e-8
    kl = torch.log(prior_width / (latent_width + eps))
    return kl.mean()


def get_uniform_logprobs(inferred_mu, inferred_width, temperatures):
    latent_dist = tdist.uniform.Uniform(
        inferred_mu - inferred_width, inferred_mu + inferred_width
    )
    cdf = latent_dist.cdf(temperatures)
    log_prob_default = latent_dist.log_prob(inferred_mu)
    probs = torch.where(
        cdf * (1 - cdf) > 0.0, log_prob_default, torch.full(cdf.shape, -8).cuda()
    )
    return probs.mean()


def get_preds_from_uniform(inferred_mu, inferred_width, categorical_temperature_prior):
    categorical_temperature_prior = torch.reshape(
        categorical_temperature_prior, [1, -1]
    )
    preds = (
        (categorical_temperature_prior > inferred_mu - inferred_width)
        * (categorical_temperature_prior < inferred_mu + inferred_width)
    ).double()
    return preds


def get_correlation(a, b):
    numerator = torch.sum((a - a.mean()) * (b - b.mean()))
    denominator = torch.sqrt(torch.sum((a - a.mean()) ** 2)) * torch.sqrt(
        torch.sum((b - b.mean()) ** 2)
    )
    return numerator / denominator


def get_offdiag_indices(num_nodes):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices
