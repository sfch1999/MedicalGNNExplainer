import os
import pickle
import numpy as np
import torch

import configs
import models
from utils import io_utils
from utils.train_utils import build_optimizer
from utils.io_utils import save_checkpoint
import time
import torch.nn as nn
import sklearn.metrics as metrics


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test


def medic(args):
    """
    Creating a simple Graph ConvNet using parameters of args (https://arxiv.org/abs/1609.02907)
    """

    # Loading DataSet from /Pickles
    global result_test, result_train
    with open('Pickles/feats.pickle', 'rb') as handle:
        feats = np.expand_dims(pickle.load(handle), axis=0)
    with open('Pickles/age_adj.pickle', 'rb') as handle:
        age_adj = pickle.load(handle)
    with open('Pickles/preds.pickle', 'rb') as handle:
        labels = np.expand_dims(pickle.load(handle), axis=0)

    # initializing model variables
    num_nodes = labels.shape[1]
    num_train = int(num_nodes * 0.9)
    num_classes = max(labels[0]) + 1
    idx = [i for i in range(num_nodes)]
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    labels = labels.astype(np.long)
    age_adj = age_adj.astype(np.float)
    feats = feats.astype(np.float)

    age_adj = age_adj + np.eye(age_adj.shape[0])
    d_hat_inv = np.linalg.inv(np.diag(age_adj.sum(axis=1))) ** (1 / 2)
    temp = np.matmul(d_hat_inv, age_adj)
    age_adj = np.matmul(temp, d_hat_inv)
    age_adj = np.expand_dims(age_adj, axis=0)

    labels_train = torch.tensor(labels[:, train_idx], dtype=torch.long)
    adj = torch.tensor(age_adj, dtype=torch.float)
    x = torch.tensor(feats, dtype=torch.float, requires_grad=True)

    # Creating a model which is used in https://github.com/RexYing/gnn-model-explainer
    model = models.GcnEncoderNode(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,
    )

    if args.gpu:
        model = model.cuda()

    scheduler, optimizer = build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    to_save = (0, None)  # used for saving best model

    # training the model
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optimizer.step()
        # for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), labels, train_idx, test_idx
        )

        if result_test["acc"] > to_save[0]:
            to_save = (result_test["acc"], (model, optimizer, args))

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )
        if epoch % 100 == 0:
            print(result_train["conf_mat"])
            print(result_test["conf_mat"])

        if scheduler is not None:
            scheduler.step()

    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    to_save[1][0].eval()
    if args.gpu:
        ypred, _ = to_save[1][0](x.cuda(), adj.cuda())
    else:
        ypred, _ = to_save[1][0](x, adj)
    cg_data = {
        "adj": age_adj,
        "feat": feats,
        "label": labels,
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }

    # saving the model so that it can be restored for GNN explaining
    print(save_checkpoint(to_save[1][0], to_save[1][1], args, num_epochs=-1, cg_dict=cg_data))

    return to_save[1][0], to_save[1][1], args, cg_data


def main():
    prog_args = configs.arg_parse()

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    return medic(prog_args)


main()
