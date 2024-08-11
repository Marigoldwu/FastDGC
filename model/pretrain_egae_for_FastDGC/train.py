# -*- coding: utf-8 -*-
"""
@Time: 2023/6/14 8:46 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam
from module.EGAE import EGAE
from utils.data_processor import normalize_adj_torch
from utils.evaluation import eva
from utils.kmeans_gpu import kmeans
from utils.result import Result
from utils.utils import get_format_variables, similarity


def train(args, data, logger):
    params_dict = {"acm": [120, 2e-3, 0, 128, 32, 3],  # 0
                   "cora": [70, 1e-3, 1, 1024, 32, 3],  # 256-32 74.08  0
                   "dblp": [25, 2e-3, 1, 256, 16, 3],
                   "cite": [10, 2e-3, 1, 1024, 32, 3],
                   "amap": [160, 1e-3, 1, 128, 32, 1],
                   "wisc": [25, 1e-2, 1, 256, 32, 1],
                   "texas": [20, 1e-3, 1000, 128, 32, 1],
                   }
    args.pretrain_epoch = params_dict[args.dataset_name][0]
    args.pretrain_lr = params_dict[args.dataset_name][1]
    args.epsilon = params_dict[args.dataset_name][2]
    args.linear_dim = params_dict[args.dataset_name][3]
    args.embedding_dim = params_dict[args.dataset_name][4]
    args.times = params_dict[args.dataset_name][5]

    pretrain_egae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = EGAE(args.input_dim, args.linear_dim, args.embedding_dim).to(args.device)
    logger.info(model)
    optimizer = Adam(model.parameters(), lr=args.pretrain_lr)

    feature = data.feature.to(args.device).float()
    adj_label = data.adj.to(args.device).float()
    adj_norm = normalize_adj_torch(adj_label)
    label = data.label

    sf = similarity(feature)

    acc_max, embedding = -1, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]

    for epoch in range(1, args.pretrain_epoch + 1):
        model.train()
        A_pred, embedding, xt = model(feature, adj_norm, args.times)
        sx = similarity(xt)
        re_loss = torch.norm(A_pred - adj_label) / args.nodes
        ce_loss = F.cross_entropy(sx, sf)
        loss = re_loss + args.epsilon * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            predict_labels, _ = kmeans(X=embedding, num_clusters=args.clusters,
                                       distance="euclidean", device="cuda")
            acc, nmi, ari, f1 = eva(label, predict_labels.numpy())

            if acc > acc_max:
                acc_max = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]

            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", loss=f"{loss.detach().cpu().numpy():0>.4f}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    torch.save(model.state_dict(), pretrain_egae_filename)
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    return result
