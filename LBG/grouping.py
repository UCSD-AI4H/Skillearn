import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from resnet import ResNet
from models import VGG11
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from types import SimpleNamespace
from fairtorch import DemographicParityLoss

import torch.optim as optim

import torchvision.models as models
# resnet18 = models.resnet18(pretrained=True)

class SparseDispatcher(object):
    def __init__(self, num_experts, gates, device='cuda'):
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        # self._part_sizes = (gates > 0).sum(0).tolist()
        self._part_sizes = (gates != 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        self.device = device

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        # print(inp_exp.size())
        # print(self._part_sizes)
        # print(self._gates)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True).to(self.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float()).to(self.device)
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()


    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, batch_size, args, lr=0.001,weight_decay=0.001,noisy_gating=False, k=1, device='cuda'):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.batch_size = batch_size
        self.lr=lr
        self.weight_decay=weight_decay
        self.args = args
        self.input_size = input_size
        self.device = device
        self.dp_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=100).to(self.device)
        self.k = k
        self.img_size = int((input_size/3)**(1/2))
        if self.args.dataset == 'isic18':
            self.experts = nn.ModuleList()
            for _ in range(self.num_experts):
                expert_model = models.resnet18(pretrained=True)
                num_features = expert_model.fc.in_features
                expert_model.fc = nn.Linear(num_features, self.output_size)
                self.experts.append(expert_model)
        else:
            self.experts = nn.ModuleList([ResNet(in_planes = 32, num_classes=self.output_size) for i in range(self.num_experts)])
            # self.experts = nn.ModuleList([VGG11(num_classes=self.output_size) for i in range(self.num_experts)])
        
        

        self.w_gate = Variable(torch.randn(input_size, num_experts).cuda(), requires_grad=True)
        with torch.no_grad():
            # initialize to smaller value
            self.w_gate.mul_(1e-2)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        
        self.normal = Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device))
        # self.register_buffer("mean", torch.tensor([0.0]))
        # self.register_buffer("std", torch.tensor([1.0]))
        
        # self.optimizer = optim.Adam(self.T(),lr=0.0001,betas=(0.5, 0.999),weight_decay=0.001,)
        # self.optimizer = optim.SGD(self.T(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        self.optimizer = optim.SGD(self.T(), lr=self.lr, momentum=0.9)
        
        
        assert(self.k <= self.num_experts)
        
    def T(self):
        return [self.w_gate]

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def top_k_gating(self, x, Cats, train):
        # print(train)
        logits = x @ self.w_gate
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        load = self._gates_to_load(gates)
        
        if train:
            # self.w_gate.grad.zero_()
            self.optimizer.zero_grad()
            G = [] 
            # Cats = self.softmax(Cats)
            # Cats = zeros.scatter(1, top_k_indices, top_k_gates)
            zeros_gt = torch.zeros_like(Cats, requires_grad=True)
            top_targets, top_indices_gt = Cats.topk(min(self.k + 1, self.num_experts), dim=1)
            top_k_targets = top_targets[:, :self.k]
            top_k_indices_gt = top_indices_gt[:, :self.k]
            top_k_gates_gt = self.softmax(top_k_targets)
            Cats = zeros_gt.scatter(1, top_k_indices_gt, top_k_gates_gt)


            for i in range(logits.size()[0]):
                for j in range(self.num_experts):
                    if Cats[i][j] == 1:
                        G.append(j)
            G = torch.Tensor(G).to(self.device)
            loss = self.criterion(logits, G.long())  
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm(self.T(), 5)
            self.optimizer.step()
        return gates, load
        

    def loss(self, x, Cats, train, target, acc=False, aux=False, sensitive_features=None):
        logits, aux_loss = self.forward(x, Cats, train)
        class_indices = torch.argmax(logits, dim=1)
        if sensitive_features is not None:
            total_loss = self.dp_loss(x, class_indices, sensitive_features,target)
        else:
            total_loss = 0
        if not acc:
            if aux:
                return total_loss + self.criterion(logits, target)+aux_loss
            else:
                return total_loss + self.criterion(logits, target)
        else:
            correct = (logits.argmax(dim=1) == target).float().sum().item()
            if aux:
                return total_loss + self.criterion(logits, target)+aux_loss, correct
            else:
                return total_loss + self.criterion(logits, target), correct

    def forward(self, x, Cats, train):
        x = x.view(x.shape[0], -1)
        # print(x.size())
        gates, load = self.top_k_gating(x, Cats, train)
        # print(load)
        
        loss_coef=1e-2
        importance = gates.sum(0)
        aux_loss = self.cv_squared(importance) + self.cv_squared(load)
        aux_loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = []
        # expert_outputs = [self.experts[i](torch.reshape(expert_inputs[i],(gates[i].size(dim=0),3,32,32))) for i in range(self.num_experts)]
        for i in range(self.num_experts):
            # print(load[i])
            if load[i]!=0:
                expert_outputs.append(self.experts[i](torch.reshape(expert_inputs[i],(gates[i].size(dim=0),3,self.img_size,self.img_size))))
                # expert_outputs = [self.experts[i](torch.reshape(expert_inputs[i],(gates[i].size(dim=0),3,32,32)))]
        y = dispatcher.combine(expert_outputs)
        torch.save(Cats, 'categories_C.pt')
        torch.save(self.w_gate, 'categories_T.pt')
        return y,aux_loss
