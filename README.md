
# TaylorKAN

Inspired by FourierKAN, we attempted to use Taylor series expansion to accomplish tasks on the MNIST dataset.
This code only accomplishes a small task, we are working on trying more complex tasks and continuously optimizing the Taylor method.
## References

This project is inspired by the FourierKAN in the following repositories:
- [FourierKAN](https://github.com/GistNoesis/FourierKAN) - This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients.

## Core

### Taylor Layer

The `TaylorLayer` class is defined to compute the Taylor series expansion up to the specified order:

```python
class TaylorLayer(nn.Module):
  def __init__(self, input_dim, out_dim, order, addbias=True):
    super(TaylorLayer, self).__init__()
    self.input_dim = input_dim
    self.out_dim = out_dim
    self.order = order
    self.addbias = addbias

    self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
    if self.addbias:
      self.bias = nn.Parameter(torch.zeros(1, out_dim))

  def forward(self, x):
    shape = x.shape
    outshape = shape[0:-1] + (self.out_dim,)
    x = torch.reshape(x, (-1, self.input_dim))
    x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

    y = torch.zeros((x.shape[0], self.out_dim), device=x.device)

    for i in range(self.order):
      term = (x_expanded ** i) * self.coeffs[:, :, i]
      y += term.sum(dim=-1)

    if self.addbias:
      y += self.bias

    y = torch.reshape(y, outshape)
    return y
```

### TaylorCNN

The `TaylorCNN` class defines the CNN with Taylor layers:

```python
class TaylorCNN(nn.Module):
  def __init__(self):
    super(TaylorCNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.pool2 = nn.MaxPool2d(2)
    self.taylorkan1 = TaylorLayer(32*7*7, 128, 2)
    self.taylorkan2 = TaylorLayer(128, 10, 2)

  def forward(self, x):
    x = F.selu(self.conv1(x))
    x = self.pool1(x)
    x = F.selu(self.conv2(x))
    x = self.pool2(x)
    x = x.view(x.size(0), -1)
    x = self.taylorkan1(x)
    x = self.taylorkan2(x)
    return x
```

### Training and Evaluation

The `train` and `evaluate` functions handle the training and evaluation processes:

```python
def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data.to(device))
    loss = nn.CrossEntropyLoss()(output, target.to(device))
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
      print(f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += nn.CrossEntropyLoss()(output, target).item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
```

### Running

To run the entire process:
```python
for epoch in range(0, 2):
  train(model, device, train_loader, optimizer, epoch)
evaluate(model, device, test_loader)
```

## License

This project is licensed under the MIT License.
