import json
import torch
import logging
import numpy as np
from tqdm import tqdm
import os.path as osp

def top_k_recall(y_true, y_pred, k):
    """
    计算 Top-k Recall。
    :param y_true: 真实标签的张量
    :param y_pred: 预测排名的张量
    :param k: 考虑的前 k 个预测
    :return: Top-k Recall 值
    """
    with torch.no_grad():
        pred_top_k = y_pred.topk(k, dim=1)[1]
        correct = (pred_top_k == y_true).sum().item()
        recall_value = correct / len(y_true)
    return recall_value

def ndcg_score(y_true, y_pred, k):
    """
    计算 Normalized Discounted Cumulative Gain (NDCG)。
    :param y_true: 真实标签的张量
    :param y_pred: 预测排名的张量
    :param k: 考虑的前 k 个预测
    :return: NDCG 值
    """
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0

    with torch.no_grad():
        pred_top_k = y_pred.topk(k, dim=1)[1].cpu().numpy()
        labels = y_true.cpu().numpy()
        scores = []
        for label, pred in zip(labels, pred_top_k):
            r = [1 if label in pred[:k] else 0]
            idcg = dcg_at_k(sorted(r, reverse=True), k)
            dcg = dcg_at_k(r, k)
            score = dcg / idcg if idcg > 0 else 0
            scores.append(score)
    return torch.tensor(scores).mean()

def mean_average_precision(y_true, y_pred, k):
    """
    计算 Mean Average Precision (MAP)。
    :param y_true: 真实标签的张量
    :param y_pred: 预测排名的张量
    :param k: 考虑的前 k 个预测
    :return: MAP 值
    """
    with torch.no_grad():
        pred_top_k = y_pred.topk(k, dim=1)[1]
        labels = y_true.cpu().numpy()
        scores = []
        for label, pred in zip(labels, pred_top_k.cpu().numpy()):
            score = 0.0
            num_hits = 0.0
            for i, p in enumerate(pred[:k]):
                if p == label:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)
            scores.append(score / min(len(labels), k))
    return torch.tensor(scores).mean()

def mean_reciprocal_rank(y_true, y_pred):
    """
    计算 Mean Reciprocal Rank (MRR)。
    :param y_true: 真实标签的张量
    :param y_pred: 预测排名的张量
    :return: MRR 值
    """
    with torch.no_grad():
        pred_rankings = torch.argsort(y_pred, dim=1, descending=True)
        ranks = (pred_rankings == y_true).nonzero(as_tuple=True)[1] + 1
        return (1.0 / ranks.float()).mean()

def save_model(model, optimizer, save_variable_list, run_args, argparse_dict):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """
    with open(osp.join(run_args.log_path, 'config.json'), 'w') as fjson:
        for key, value in argparse_dict.items():
            if isinstance(value, torch.Tensor):
                argparse_dict[key] = value.numpy().tolist()
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        osp.join(run_args.save_path, 'checkpoint.pt')
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_step(model, data, ks=(1, 5, 10, 20)):
    model.eval()
    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for row in tqdm(data):
            split_index = torch.max(row.adjs_t[1].storage.row()).tolist()
            row = row.to(model.device)

            input_data = {
                'x': row.x,
                'edge_index': row.adjs_t,
                'edge_attr': row.edge_attrs,
                'split_index': split_index,
                'delta_ts': row.edge_delta_ts,
                'delta_ss': row.edge_delta_ss,
                'edge_type': row.edge_types
            }

            out, loss = model(input_data, label=row.y[:, 0], mode='test')
            loss_list.append(loss.cpu().detach().numpy().tolist())
            ranking = torch.sort(out, descending=True)[1]
            pred_list.append(ranking.cpu().detach())
            label_list.append(row.y[:, :1].cpu())
    pred_ = torch.cat(pred_list, dim=0)
    label_ = torch.cat(label_list, dim=0)
    recalls, NDCGs, MAPs = {}, {}, {}
    logging.info(f"[Evaluating] Average loss: {np.mean(loss_list)}")
    for k_ in ks:
        recalls[k_] = top_k_recall(label_, pred_, k_)
        logging.info(f"[Evaluating]CorrectIndex@{k_}:{torch.where(label_==pred_[:, :k_])[0].tolist()}")
        NDCGs[k_] = ndcg_score(label_, pred_, k_)
        MAPs[k_] = mean_average_precision(label_, pred_, k_)
        print(f'{recalls[k_]}')
        logging.info(f"[Evaluating] Recall@{k_} : {recalls[k_]},\tNDCG@{k_} : {NDCGs[k_]},\tMAP@{k_} : {MAPs[k_]}")
    mrr_res = mean_reciprocal_rank(label_, pred_).cpu().detach().numpy().tolist()
    logging.info(f"[Evaluating] MRR : {mrr_res}")
    return recalls, NDCGs, MAPs, mrr_res, np.mean(loss_list)
