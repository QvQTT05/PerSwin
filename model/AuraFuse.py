import torch
import torch.nn as nn
import torch.nn.functional as F
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        x = x * self.weight + self.bias
        return x


class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x):
        return F.prelu(x, self.alpha)


class AuraFuse(nn.Module):
    def __init__(self):
        super(AuraFuse, self).__init__()
        self.relu_act = PReLU()

    def forward(self, feature_a, feature_b):
        cosine_similarity = F.cosine_similarity(feature_a, feature_b, dim=1)
        cosine_similarity = cosine_similarity.unsqueeze(1)
        feature_a = feature_a + self.relu_act(feature_b * cosine_similarity)
        return feature_a