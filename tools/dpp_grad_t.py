import torch
import os

import matplotlib.pyplot as plt
import random
from collections import deque
import numpy as np


def scatter(data, count=0, suffix=''):
    data = data.detach().cpu().numpy()
    # for i in range(data.shape[0]):
    #     print(data[i].detach().cpu().numpy())
    plt.cla()
    plt.scatter(data[:, 0], data[:, 1])

    plt.xlim(left=-1.2, right=1.2)
    plt.ylim(top=1.2, bottom=-1.2)
    plt.title(f'step: {count}{suffix}')

def get_rbf_matrix_pre(data, centers, alpha):
    out_shape = torch.Size([data.shape[0], centers.shape[0], data.shape[-1]])
    data = data.unsqueeze(1).expand(out_shape)
    centers = centers.unsqueeze(0).expand(out_shape)
    mtx = (-(centers - data).pow(2) * alpha).sum(dim=-1, keepdim=False)
    # mtx = mtx.clamp_min(mtx.min().item() * 1)
    return mtx

def get_rbf_matrix(data, centers, alpha):
    out_shape = torch.Size([data.shape[0], centers.shape[0], data.shape[-1]])
    data = data.unsqueeze(1).expand(out_shape)
    centers = centers.unsqueeze(0).expand(out_shape)
    mtx = (-(centers - data).pow(2) * alpha).sum(dim=-1, keepdim=False).exp()
    # mtx = mtx.clamp_min(mtx.min().item() * 1)
    return mtx

def get_loss_dpp_pinverse(y):
    # K = (y.matmul(y.t()) - 1).exp() + torch.eye(y.shape[0]) * 1e-3
    K = get_rbf_matrix(y, y, 2) + torch.eye(y.shape[0], device=y.device) * 1e-5
    with torch.no_grad():
        pinverse = torch.pinverse(K.detach()).detach()
    loss = -(K * pinverse).sum()
    # loss = -torch.logdet(K)
    # loss = -(y.pinverse().t().detach() * y).sum()
    return loss


def get_loss_dpp(y, kernel='rbf'):
    # K = (y.matmul(y.t()) - 1).exp() + torch.eye(y.shape[0]) * 1e-3
    if kernel == 'rbf':
        K = get_rbf_matrix(y, y, 1.0) + torch.eye(y.shape[0], device=y.device) * 1e-5
    elif kernel == 'inner':
        # y = y / y.pow(2).sum(dim=-1, keepdim=True).sqrt()
        K = y.matmul(y.t()).exp()
        # K = torch.softmax(K, dim=0)
        K = K + torch.eye(y.shape[0], device=y.device) * 1e-4
        print(K)
        # print('k shape: ', K.shape, ', y_mtx shape: ', y_mtx.shape)
    else:
        assert False
    loss = -torch.logdet(K)
    # loss = -(y.pinverse().t().detach() * y).sum()
    return loss

def get_loss_cov(y):
    cov = (y - y.mean(dim=0, keepdim=True)).pow(2).mean()
    return -cov

class contrastive_loss:
    def __init__(self, dim, device=torch.device('cpu')):
        self.device = device
        self.W = torch.rand((dim, dim), requires_grad=True, device=device)
        # self.W = torch.eye(dim, requires_grad=True, device=device)
        self.w_optim = torch.optim.Adam([self.W], lr=1e-1)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def get_loss_meta(self, y, need_w_grad=False):
        W = self.W
        if need_w_grad:
            proj_k = (W).matmul(y.t())
        else:
            proj_k = (W).detach().matmul(y.t())
        # proj_k = 30 * torch.eye((self.W + self.W.t()).shape[0]).detach().matmul(y.t())
        # proj_k = y.t()
        print(self.W)
        # logits = y.matmul(proj_k)
        logits = get_rbf_matrix(y, y, 10.0)
        # print(logits.max(dim=1, keepdim=True).values)
        logits = logits - logits.max(dim=1, keepdim=True).values
        # logits = get_rbf_matrix(y, y, alpha=1)
        labels = torch.arange(logits.shape[0])
        # print(logits)
        loss = self.loss_func(logits, labels)
        return loss

    def get_loss(self, y):
        # loss_w = self.get_loss_meta(y.detach(), True)
        # self.w_optim.zero_grad()
        # loss_w.backward()
        # self.w_optim.step()
        return self.get_loss_meta(y)

def make_seq():
    return torch.nn.Sequential(
        torch.nn.Linear(intput_len, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, output_len),
        torch.nn.Tanh()
    )


def copy_from_net(net_src, net_target, tau=0.99):
    with torch.no_grad():
        for param, target_param in zip(net_src.parameters(True), net_target.parameters(True)):
            target_param.data.mul_(tau)
            target_param.data.add_((1 - tau) * param.data)


if __name__ == '__main__':
    import seaborn as sen
    sen.set_theme()
    intput_len = 24
    output_len = 2
    torch.manual_seed(3)
    net = make_seq()
    net_target = make_seq()
    optim = torch.optim.Adam(net.parameters(True), lr=3e-4)
    contra_loss = contrastive_loss(output_len)
    x = torch.randn((40, intput_len))
    plt.ion()
    lst_y = deque(maxlen=3)
    # y_mean = None
    name = ', determinant'
    if not os.path.exists('pics'):
        os.makedirs('pics')
    for i in range(10000):
        copy_from_net(net, net_target, 0.99)
        y = net(x)
        # y_mean = net_target(x)
        # if y_mean is None:
        #     y_mean = y
        # else:
        #     y_mean = 0.99 * y_mean.detach() + 0.01 * y
        lst_y.append(y.detach())
        # y = y / y.pow(2).sum(-1, keepdim=True).sqrt()
        scatter(y, i, name)
        loss_dpp = get_loss_dpp(y, 'rbf')
        loss_cov = get_loss_cov(y)
        loss_ct = contra_loss.get_loss(y)
        loss = loss_dpp  # loss_dpp # loss_cov  #  + (y - lst_y[0]).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_([*net.parameters(True)], 5)
        norm = 0
        optim.step()
        print(f"{i}: loss: {loss.item()}, loss_dpp: {loss_dpp.item()}, loss_cov: {loss_cov.item()}, loss_ct: {loss_ct.item()}, norm: {norm}")
        # if i == 0:
        #     plt.pause(1.0)  # 启动时间，方便截屏
        plt.pause(0.01)
        # plt.title('loss: {:.2f}'.format(loss.item()), fontsize=15)
        # plt.title('')
        if i % 20 == 0:
            plt.savefig(os.path.join('pics', 'pic_{}.pdf'.format(i)), bbox_inches='tight')
    plt.ioff()
    plt.show()
    pass
