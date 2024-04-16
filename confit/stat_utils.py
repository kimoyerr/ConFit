import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
import torch.distributed as dist

import disco


def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]


def compute_stat(sr):
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    return mean, std


def compute_score(model, seq, mask, wt, pos, tokenizer):
    '''
    compute mutational proxy using masked marginal probability
    :param seq:mutant seq
    :param mask:attention mask for input seq
    :param wt: wild type sequence
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = seq.device

    mask_seq = seq.clone()
    m_id = tokenizer.mask_token_id

    batch_size = int(seq.shape[0])
    for i in range(batch_size):
        mut_pos = pos[i]
        mask_seq[i, mut_pos+1] = m_id

    out = model(mask_seq, mask, output_hidden_states=True)
    logits = out.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size)
    # scores = scores.to(device)

    for i in range(batch_size):

        mut_pos = pos[i]
        score_i = log_probs[i]
        wt_i = wt[i]
        seq_i = seq[i]
        # TODO: Does this compare across batches on different GPUs? If not, we need to fix this
        scores[i] = torch.sum(score_i[mut_pos+1, seq_i[mut_pos+1]])-torch.sum(score_i[mut_pos+1, wt_i[mut_pos+1]])

    return scores, logits, out, log_probs


def BT_loss(scores, golden_scores):
    loss = torch.tensor(0.)
    loss = loss.cuda()
    
    total_comparisons = 0
    for i in range(len(scores)):
        for j in range(len(scores)):
            total_comparisons += 1
            if golden_scores[i] > golden_scores[j]:
                loss += torch.log(1+torch.exp(scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-scores[j]))
    return loss


# This function is useful for understanding how to implement gather from all GPUs but suffers from memory issues with activations not being removed from memory after every forward pass
def BT_loss_disco(gpu, scores, golden_scores):
    # Gather all scores from all GPUs
    world_size = dist.get_world_size()
    gathered_scores = [
        torch.zeros_like(scores) for _ in range(world_size)
    ]
    gathered_golden_scores = [
        torch.zeros_like(golden_scores) for _ in range(world_size)
    ]
    batch_size = scores.shape[0]

    dist.all_gather(gathered_scores, scores)
    dist.all_gather(gathered_golden_scores, golden_scores)
    gathered_scores = torch.cat(gathered_scores)
    gathered_golden_scores = torch.cat(gathered_golden_scores)
    all_scores = gathered_scores.requires_grad_(True)
    all_golden_scores = gathered_golden_scores

    # Slice the scores and golden scores for the current GPU
    scores = all_scores[batch_size*gpu:batch_size*(gpu+1)]
    golden_scores = all_golden_scores[batch_size*gpu:batch_size*(gpu+1)]

    loss = torch.tensor(0.)
    loss = loss.cuda()
    total_comparisons = 0
    for i in range(len(scores)):
        for j in range(len(all_scores)):
            # print(i, j, scores[i], all_scores[j])
            total_comparisons += 1
            if golden_scores[i] > all_golden_scores[j]:
                loss += torch.log(1+torch.exp(all_scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-all_scores[j]))
    
    # Clean up
    gathered_golden_scores = None
    gathered_scores = None

    return loss, scores, all_scores, all_golden_scores


# Contrasts against all other sequences in the batch from all GPUs
# This version works with removing activations from memory compared to the function above BT_loss_disco which doesn't
def BT_loss_disco_gather(gpu, scores, golden_scores):
    # Gather all scores from all GPUs
    all_scores = disco.Gather(scores)
    all_golden_scores = disco.Gather(golden_scores)

    # Slice the scores and golden scores for the current GPU
    batch_size = scores.shape[0]
    scores = all_scores[batch_size*gpu:batch_size*(gpu+1)]
    golden_scores = all_golden_scores[batch_size*gpu:batch_size*(gpu+1)]

    loss = torch.tensor(0.)
    loss = loss.cuda()
    total_comparisons = 0
    for i in range(len(scores)):
        for j in range(len(all_scores)):
            # print(i, j, scores[i], all_scores[j])
            total_comparisons += 1
            if golden_scores[i] > all_golden_scores[j]:
                loss += torch.log(1+torch.exp(all_scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-all_scores[j]))
    print(f"Total comparisons: {total_comparisons}")
    return loss, scores, all_scores, all_golden_scores


def KLloss(logits, logits_reg, seq, att_mask):

    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = int(seq.shape[0])

    loss = torch.tensor(0.)
    loss = loss.cuda()
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]


        seq_len = torch.sum(att_mask[i])

        reg = probs_reg_i[torch.arange(0, seq_len), seq[i, :seq_len]]
        pred = probs_i[torch.arange(0, seq_len), seq[i, :seq_len]]

        # TODO: Should the inputs be reversed here? See the definition of KLDivLoss from torch.nn and make sure we are using the reference distribution in the denominator
        # loss += creterion_reg(reg.log(), pred)  # This is the original loss as implemented in the code for ConFit
        loss += creterion_reg(pred.log(), reg)
    return loss