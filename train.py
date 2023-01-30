from tqdm import tqdm
import torch.nn.functional as F

class Train():
  def __init__(self):
    self.train_losses = []
    self.train_acc = []

  def train_model(self,model, device, train_loader, optimizer, epoch, L1=False):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    lambda_l1=0
    for batch_idx, (data, target) in enumerate(pbar):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      y_pred = model(data)
      loss = F.nll_loss(y_pred, target)
      l1=0
      if L1:
        for p in model.parameters():
          l1 = l1 + p.abs().sum()
      loss = loss + lambda_l1*l1
      self.train_losses.append(loss)
      loss.backward()
      optimizer.step()
      pred = y_pred.argmax(dim=1, keepdim = True)
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed+=len(data)

      pbar.set_description(desc=f'Loss={loss.item()} Batch_id = {batch_idx} Accuracy = {100*correct/processed:0.2f}')
      self.train_acc.append(100*correct/processed)