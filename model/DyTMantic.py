import torch
import torch.nn as nn
import torch.nn.functional as F
class DyTMantic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.project1 = nn.Linear(in_dim, in_dim)
        self.nonlinear = F.gelu
        self.processed_signal = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = ManticPercept(in_dim)
        self.norm = DynamicTanh(normalized_shape=in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        self.fusion_feature1 = AuraFuse()

    def forward(self, x, hw_shapes=None):
        identity = x
        x_normed = self.norm(x) * self.gamma + x * self.gammax  # (B, N, C)
        project1 = self.project1(x_normed)
        b, n, c = project1.shape
        h, w = hw_shapes
        project1_2d = project1.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        project1_2d_processed = self.adapter_conv(project1_2d)
        project1_3d_processed = project1_2d_processed.permute(0, 2, 3, 1).contiguous().view(b, n, c)
        nonlinear = self.nonlinear(project1_3d_processed)
        nonlinear = self.dropout(nonlinear)
        project2 = self.processed_signal(nonlinear)
        return self.fusion_feature1(identity, project2)