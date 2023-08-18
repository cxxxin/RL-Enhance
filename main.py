import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Generator, Discriminator
    
# 初始化环境和网络
state_dim = 512  # 图像的像素维度
action_dim = 2  # 动作维度，描述智能体（agent）可以采取的不同动作的数量
G = Generator(conv_dim=32, norm_fun='InstanceNorm', act_fun='LeakyReLU', use_sn=False)
D = Discriminator(conv_dim=32, norm_fun='none', act_fun='LeakyReLU', use_sn=True, adv_loss_type='rals')

PolicyNet = PolicyNet(state_dim, action_dim)
ValueNet = ValueNet(state_dim, action_dim)
PolicyNet_optimizer = optim.Adam(PolicyNet.parameters(), lr=0.001)
ValueNet_optimizer = optim.Adam(ValueNet.parameters(), lr=0.001)

num_epochs = 100
gamma = 0.99
# 训练
for epoch in range(num_epochs):
    # 获取当前状态
    state = np.random.rand(state_dim)  # 这里用随机状态代替图像
    
    # PolicyNet选择动作
    action = PolicyNet(torch.tensor(state, dtype=torch.float32)).detach().numpy()
    
    # 执行动作，获取奖励和下一个状态
    next_state = np.random.rand(state_dim)  # 这里用随机状态代替图像
    reward = np.random.rand()
    
    # 更新ValueNet
    target = reward + gamma * ValueNet(torch.tensor(next_state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32))
    ValueNet_loss = nn.MSELoss()(ValueNet(torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)), target)
    ValueNet_optimizer.zero_grad()
    ValueNet_loss.backward()
    ValueNet_optimizer.step()
    
    # 更新PolicyNet
    PolicyNet_loss = -ValueNet(torch.tensor(state, dtype=torch.float32), PolicyNet(torch.tensor(state, dtype=torch.float32))).mean()
    PolicyNet_optimizer.zero_grad()
    PolicyNet_loss.backward()
    PolicyNet_optimizer.step()
