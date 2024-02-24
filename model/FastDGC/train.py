# -*- coding: utf-8 -*-
"""
@Time: 2023/6/8 15:16
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from model.FastDGC.model import FastDGC
from utils import data_processor
from utils.data_processor import normalize_adj_torch
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables, count_parameters, similarity


def train(args, data, logger):
    # epoch, lr, linear, embedding, pre-training filtering times, fine-tuning filtering times, threshold, update interval
    params_dict = {"acm": [50, 3e-3, 128, 32, 3, 1, 0.5, 2],
                   "cora": [50, 2e-3, 1024, 32, 3, 3, 0.1, 1],
                   "dblp": [50, 2e-2, 256, 16, 3, 1, 0, 10],
                   "cite": [50, 2e-3, 1024, 32, 3, 3, 0.1, 2],
                   "amap": [50, 5e-4, 128, 32, 1, 2, 0, 0],
                   }
    args.max_epoch = params_dict[args.dataset_name][0]
    args.lr = params_dict[args.dataset_name][1]
    args.linear_dim = params_dict[args.dataset_name][2]
    args.embedding_dim = params_dict[args.dataset_name][3]
    args.pre_times = params_dict[args.dataset_name][4]
    args.times = params_dict[args.dataset_name][5]
    args.threshold = params_dict[args.dataset_name][6]
    args.gap = params_dict[args.dataset_name][7]

    pretrain_egae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = FastDGC(args.input_dim, args.linear_dim, args.embedding_dim, args.clusters).to(args.device)
    model.egae.load_state_dict(torch.load(pretrain_egae_filename, map_location='cpu'))
    logger.info(model)
    optimizer = Adam(model.parameters(), lr=args.lr)

    feature = data.feature.to(args.device).float()
    adj_origin = data.adj.to(args.device).float()
    adj_norm = normalize_adj_torch(adj_origin)
    adj_label = adj_origin
    label = data.label

    with torch.no_grad():
        model.eval()
        _, embedding, _ = model.egae(feature, adj_norm, args.pre_times)

    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(embedding.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    acc_max = -1
    max_acc_corresponding_metrics = [0, 0, 0, 0]

    for epoch in range(1, args.max_epoch+1):
        model.train()
        A_cons, embedding, x, q = model(feature, adj_norm, args.times)

        with torch.no_grad():
            model.eval()
            pred = q.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(label, pred)
            if acc > acc_max:
                acc_max = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
            if args.gap:
                if epoch % args.gap == 0:
                    sx = similarity(x)
                    sx[sx <= args.threshold] = 0
                    A = torch.mul(sx, adj_origin)
                    adj_norm = normalize_adj_torch(A)
        p = data_processor.target_distribution(q.data)
        re_loss = torch.norm(A_cons - adj_label) / args.nodes
        ce_loss = F.cross_entropy(q.log(), p)
        loss = re_loss + ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info(
            get_format_variables(epoch=f"{epoch:0>3d}", loss=f"{loss.detach().cpu().numpy():0>.4f}", acc=f"{acc*100:0>.2f}",
                                 nmi=f"{nmi*100:0>.2f}",
                                 ari=f"{ari*100:0>.2f}", f1=f"{f1*100:0>.2f}"))
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The max memory allocated to model is: {mem_used:.2f} MB.")
    return result
