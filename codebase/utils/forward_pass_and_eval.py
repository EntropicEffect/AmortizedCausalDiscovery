from __future__ import division
from __future__ import print_function

from collections import defaultdict
import time
import torch

import numpy as np

from model.modules import *
from model import utils, utils_unobserved


def forward_pass_and_eval(
    args,
    encoder,
    decoder,
    data,
    relations,
    rel_rec,
    rel_send,
    hard,
    data_encoder=None,
    data_decoder=None,
    testing=False,
):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))

    #################### INPUT DATA ####################
    diff_data_enc_dec = False
    if data_encoder is not None and data_decoder is not None:
        diff_data_enc_dec = True

    if data_encoder is None:
        data_encoder = data
    if data_decoder is None:
        data_decoder = data

    #################### DATA WITH UNOBSERVED TIME-SERIES ####################
    predicted_atoms = args.num_atoms
    if args.unobserved > 0:
        if args.shuffle_unobserved:
            mask_idx = np.random.randint(0, args.num_atoms)
        else:
            mask_idx = args.num_atoms - 1

        ### baselines ###
        if args.model_unobserved == 1:
            (
                data_encoder,
                data_decoder,
                predicted_atoms,
                relations,
            ) = utils_unobserved.baseline_remove_unobserved(
                args, data_encoder, data_decoder, mask_idx, relations, predicted_atoms
            )
            unobserved = 0
        if args.model_unobserved == 2:
            (
                data_encoder,
                unobserved,
                losses["mse_unobserved"],
            ) = utils_unobserved.baseline_mean_imputation(args, data_encoder, mask_idx)
            data_decoder = utils_unobserved.add_unobserved_to_data(
                args, data_decoder, unobserved, mask_idx, diff_data_enc_dec
            )
    else:
        mask_idx = 0
        unobserved = 0

    #################### ENCODER ####################
    if args.use_encoder:
        if args.unobserved > 0 and args.model_unobserved == 0:
            # model unobserved time-series
            (
                logits,
                unobserved,
                losses["mse_unobserved"],
            ) = encoder(data_encoder, rel_rec, rel_send, mask_idx=mask_idx)
            data_decoder = utils_unobserved.add_unobserved_to_data(
                args, data_decoder, unobserved, mask_idx, diff_data_enc_dec
            )
        else:
            # model only the edges
            logits = encoder(data_encoder, rel_rec, rel_send)

    edges = logits  # utils.gumbel_softmax(logits, tau=args.temp, hard=hard)
    prob = utils.my_softmax(logits, -1)

    target = data_decoder[:, :, 1:, :]

    #################### DECODER ####################
    if args.decoder == "rnn":
        output = decoder(
            data_decoder,
            edges,
            rel_rec,
            rel_send,
            pred_steps=args.prediction_steps,
            burn_in=True,
            burn_in_steps=args.timesteps - args.prediction_steps,
        )
    else:
        output = decoder(
                data_decoder,
                edges,
                rel_rec,
                rel_send,
                args.prediction_steps,
            )

    #################### LOSSES ####################
    if args.unobserved > 0:
        if args.model_unobserved != 1:
            losses["mse_observed"] = utils_unobserved.calc_mse_observed(
                args, output, target, mask_idx
            )

            if not args.shuffle_unobserved:
                losses["observed_acc"] = utils.edge_accuracy_observed(
                    logits, relations, num_atoms=args.num_atoms
                )

    ## calculate performance based on how many particles are influenced by unobserved one/last one
    if not args.shuffle_unobserved and args.unobserved > 0:
        losses = utils_unobserved.calc_performance_per_num_influenced(
            args,
            relations,
            output,
            target,
            logits,
            prob,
            mask_idx,
            losses
        )

    #################### MAIN LOSSES ####################
    ### latent losses ###
    losses["acc"] = utils.edge_accuracy(logits, relations)

    ### output losses ###

    losses["loss_mse"] = F.mse_loss(output, target)

    total_loss = 0  # FIXME
    total_loss += args.teacher_forcing * losses["mse_unobserved"]
    losses["loss"] = total_loss

    losses["inference time"] = time.time() - start

    return losses, output, unobserved, edges
