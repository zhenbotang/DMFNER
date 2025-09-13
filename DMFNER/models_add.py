import torch
import torch.nn as nn


class DDZAttention(nn.Module):

    def __init__(self, channel=1024, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(in_features=channel, out_features=channel)
        self.fc2 = nn.Linear(in_features=channel, out_features=channel)
        self.fc3 = nn.Linear(in_features=channel, out_features=channel)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 3 * channel, bias=False),
        )
        self.Sigmoid_1 = nn.Sigmoid()
        self.Sigmoid_2 = nn.Sigmoid()
        self.Sigmoid_3 = nn.Sigmoid()
        self.channel = channel

    def forward(self, x):
        B, N, C = x.size()
        linear_outs = []
        x1 = self.fc1(x).permute(0, 2, 1)
        linear_outs.append(x1)
        x2 = self.fc2(x).permute(0, 2, 1)
        linear_outs.append(x2)
        x3 = self.fc3(x).permute(0, 2, 1)
        linear_outs.append(x3)
        add = sum(linear_outs)  # (B,C,N)-->(B,C,N)
        pool = add.mean(-1)

        y = self.fc(pool)
        y_split_tensors = torch.split(y, split_size_or_sections=self.channel, dim=1)
        y1 = self.Sigmoid_1(y_split_tensors[0]).view(B, C, 1) * x1
        y2 = self.Sigmoid_2(y_split_tensors[1]).view(B, C, 1) * x2
        y3 = self.Sigmoid_3(y_split_tensors[2]).view(B, C, 1) * x3
        y = y1 + y2 + y3
        y = y.permute(0, 2, 1)

        return y


class EfficientAdditiveAttnetion(nn.Module):

    def __init__(self, in_dims=1024, token_dim=1024):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim)
        self.to_key = nn.Linear(in_dims, token_dim)

        self.w_a = nn.Parameter(torch.randn(token_dim, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim, token_dim)
        self.final = nn.Linear(token_dim, token_dim)

    def forward(self, x):
        B,N,D = x.shape

        query = self.to_query(x)
        key = self.to_key(x)

        # 进行标准化
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        # 学习query的注意力权重(1,50,1)
        query_weight = query @ self.w_a
        A = query_weight * self.scale_factor
        A = torch.nn.functional.normalize(A, dim=1)

        q = torch.sum(A * query, dim=1)
        # q = einops.repeat(q, "b d -> b repeat d", repeat=key.shape[1]) # BxNxD
        q = q.reshape(B, 1, -1)

        out = self.Proj(q * key) + query
        out = self.final(out)

        return out


class MSFblock(nn.Module):
    def __init__(self, in_channels):
        super(MSFblock, self).__init__()
        out_channels = in_channels

        self.project = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=2)
        self.Sigmoid = nn.Sigmoid()
        self.SE1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0, dilation=1)
        self.SE2 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0, dilation=1)

    def forward(self, x0, x1):
        # x1/x2: (B,N,C)
        # 8,60,1024 -> 8,1024,60
        y0 = x0.permute(0, 2, 1)
        y1 = x1.permute(0, 2, 1)

        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))

        weight = torch.cat([y0_weight, y1_weight], 2)

        weight = self.softmax(self.Sigmoid(weight))


        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)


        x_att = y0_weight * y0 + y1_weight * y1

        return self.project(x_att).permute(0, 2, 1)


class MSFblock3d(nn.Module):
    def __init__(self, in_channels):
        super(MSFblock3d, self).__init__()
        out_channels = in_channels

        self.project = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=2)
        self.Sigmoid = nn.Sigmoid()
        self.SE1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0, dilation=1)
        self.SE2 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0, dilation=1)
        self.SE3 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0, dilation=1)

    def forward(self, x0, x1, x2):
        # x1/x2: (B,N,C)
        # 8,60,1024 -> 8,1024,60
        y0 = x0.permute(0, 2, 1)
        y1 = x1.permute(0, 2, 1)
        y2 = x2.permute(0, 2, 1)

        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))

        weight = torch.cat([y0_weight, y1_weight, y2_weight], 2)

        weight = self.softmax(self.Sigmoid(weight))

        # weight[:,:,0]:(B,C); (B,C)-->unsqueeze-->(B,C,1)
        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)


        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2

        return self.project(x_att).permute(0, 2, 1)


if __name__ == '__main__':
    input1 = torch.randn(8, 60, 1024)
    Model = DDZAttention(channel=1024, reduction=8)
    output = Model(input1)
    print(output.shape)
