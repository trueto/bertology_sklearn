#   Copyright 2020 trueto

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# refer to: https://github.com/yatengLG/Focal-Loss-Pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F

class Focal_Loss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2,
                 num_classes=2, size_average=True):
        """
       focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
       步骤详细的实现了 focal_loss损失函数.
       :param alpha:   阿尔法α,类别权重.
       当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],
       常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
       :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
       :param num_classes:     类别数量
       :param size_average:    损失计算方式,默认取均值
       """

        super().__init__()
        self.size_average = size_average

        if isinstance(alpha, list):
            # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)

        else:
            # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            assert alpha < 1
            # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C] 分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """

        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1, labels.view(-1,1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss