import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, weight=None, ignore_index: int = -1, smooth: float = 1.0):
        super().__init__()
        self.C            = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid   = targets != self.ignore_index
        logits  = logits[valid]             
        targets = targets[valid]          

        probs   = F.softmax(logits, dim=1)  
        one_hot = F.one_hot(targets, self.C).float()

        inter  = (probs * one_hot).sum(0)    
        denom  = probs.sum(0) + one_hot.sum(0)
        dice_c = (2 * inter + self.smooth) / (denom + self.smooth) 

        if self.weight is not None:
            return 1.0 - (dice_c * self.weight).sum() / self.weight.sum()
        return 1.0 - dice_c.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None, ignore_index=-1, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none', 
            weight=self.weight, 
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )        
        pt = torch.exp(-ce_loss)
        f_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return f_loss.mean()
        elif self.reduction == 'sum':
            return f_loss.sum()
        else:
            return f_loss

class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes:     int,
        gamma: float = 2.0,
        alpha: float = 0.5,     
        weight: torch.Tensor = None,
        ignore_index:    int = -1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(
            gamma = gamma,
            weight = weight,
            ignore_index = ignore_index,
            label_smoothing = label_smoothing,
        )
        self.dice = MulticlassDiceLoss(
            num_classes = num_classes,
            weight = weight,
            ignore_index = ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.alpha) * self.focal(logits, targets) \
             +        self.alpha  * self.dice(logits,  targets)