import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.
	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.
	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes=0, epsilon=0.1, topk_smoothing=False):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
		self.k = 1 if not topk_smoothing else self.num_classes//50

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		if self.k >1:
			topk = torch.argsort(-log_probs)[:,:self.k]
			targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
			targets += torch.zeros_like(log_probs).scatter_(1, topk, self.epsilon / self.k)
		else:
			targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
			targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss


class SoftEntropy(nn.Module):
	def __init__(self, input_prob=False):
		super(SoftEntropy, self).__init__()
		self.input_prob = input_prob
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		if self.input_prob:
			loss = (- targets.detach() * log_probs).mean(0).sum()
		else:
			loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
		return loss


class SoftEntropySmooth(nn.Module):
	def __init__(self, epsilon=0.1):
		super(SoftEntropySmooth, self).__init__()
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, soft_targets, targets):
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		soft_targets = F.softmax(soft_targets, dim=1)
		smooth_targets = (1 - self.epsilon) * targets + self.epsilon * soft_targets
		loss = (- smooth_targets.detach() * log_probs).mean(0).sum()
		return loss
	

class Softmax(nn.Module):

	def __init__(self, feat_dim, num_class, temp=0.05):
		super(Softmax, self).__init__()
		self.weight = Parameter(torch.Tensor(feat_dim, num_class))
		self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
		self.temp = temp

	def forward(self, feats, labels):
		kernel_norm = F.normalize(self.weight, dim=0)
		feats = F.normalize(feats)
		outputs = feats.mm(kernel_norm)
		outputs /= self.temp
		loss = F.cross_entropy(outputs, labels)
		return loss


class CircleLoss(nn.Module):
    """Implementation for "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    Note: this is the classification based implementation of circle loss.
    """
    def __init__(self, feat_dim, num_class, margin=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.gamma = gamma

        self.O_p = 1 + margin
        self.O_n = -margin
        self.delta_p = 1-margin
        self.delta_n = margin

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        index_pos = torch.zeros_like(cos_theta)        
        index_pos.scatter_(1, labels.data.view(-1, 1), 1)
        index_pos = index_pos.bool()
        index_neg = torch.ones_like(cos_theta)        
        index_neg.scatter_(1, labels.data.view(-1, 1), 0)
        index_neg = index_neg.bool()

        alpha_p = torch.clamp_min(self.O_p - cos_theta.detach(), min=0.)
        alpha_n = torch.clamp_min(cos_theta.detach() - self.O_n, min=0.)

        logit_p = alpha_p * (cos_theta - self.delta_p)
        logit_n = alpha_n * (cos_theta - self.delta_n)

        output = cos_theta * 1.0
        output[index_pos] = logit_p[index_pos]
        output[index_neg] = logit_n[index_neg]
        output *= self.gamma

        return F.cross_entropy(output, labels)


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta)-m
    """
    def __init__(self, feat_dim, num_class, s = 64.0, m = 0.35):
        super(CosFace, self).__init__()
        self.in_features = feat_dim
        self.out_features = num_class
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(feat_dim, num_class))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight, dim=1))
        cosine = torch.mm(F.normalize(input), F.normalize(self.weight, dim=0)) 
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device = 'cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return F.cross_entropy(output, label)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'


import math

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        print('Initializing FocalLoss for training: alpha={}, gamma={}'.format(self.alpha, self.gamma))

    def forward(self, input, target):
        assert input.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1

        logpt = F.log_softmax(input, dim=1)
        logpt_gt = logpt.gather(1,target.unsqueeze(1))
        logpt_gt = logpt_gt.view(-1)
        pt_gt = logpt_gt.exp()
        assert logpt_gt.size() == pt_gt.size()
        
        loss = -self.alpha*(torch.pow((1-pt_gt), self.gamma))*logpt_gt
        
        return loss.mean()


class LabelRefineLoss(nn.Module):
    def __init__(self, lambda1=0.0):
        super(LabelRefineLoss, self).__init__()
        self.lambda1 = lambda1
        print('Initializing LabelRefineLoss for training: lambda1={}'.format(self.lambda1))
            
    def forward(self, input, target):
        assert input.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1

        logpt = F.log_softmax(input, dim=1)
        logpt_gt = logpt.gather(1,target.unsqueeze(1))
        logpt_gt = logpt_gt.view(-1)
        logpt_pred,_ = torch.max(logpt,1)
        logpt_pred = logpt_pred.view(-1)
        assert logpt_gt.size() == logpt_pred.size()
        loss = - (1-self.lambda1)*logpt_gt - self.lambda1* logpt_pred
        
        return loss.mean()


class FocalTopLoss(nn.Module):
    def __init__(self, top_percent=0.7):
        super(FocalTopLoss, self).__init__()
        self.top_percent = top_percent

    def masked_softmax_multi_focal(self, vec, targets=None, dim=1):
        exps = torch.exp(vec)
        one_hot_pos = F.one_hot(targets, num_classes=exps.shape[1])

        one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)
        one_hot_neg = one_hot_neg - one_hot_pos
        
        neg_exps = exps.new_zeros(size=exps.shape)
        neg_exps[one_hot_neg>0] = exps[one_hot_neg>0]
        ori_neg_exps = neg_exps
        neg_exps = neg_exps/neg_exps.sum(dim=1, keepdim=True)
        
        new_exps = exps.new_zeros(size=exps.shape)
        new_exps[one_hot_pos>0] = exps[one_hot_pos>0]

        sorted, indices = torch.sort(neg_exps, dim=1, descending=True)
        sorted_cum_sum = torch.cumsum(sorted, dim=1)
        sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs()
        sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)
        
        min_values = sorted[torch.range(0, sorted.shape[0]-1).long(), sorted_cum_min_indices]
        min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)
        ori_neg_exps[ori_neg_exps<min_values] = 0

        new_exps[one_hot_neg>0] = ori_neg_exps[one_hot_neg>0]

        masked_sums = exps.sum(dim, keepdim=True)
        return new_exps / masked_sums

    def forward(self, input, target):
        masked_sim = self.masked_softmax_multi_focal(input, target)
        return F.nll_loss(torch.log(masked_sim + 1e-6), target)