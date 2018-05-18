## Pytorch API

#### import torch

TODO: numpy cpu 和 gpu,cuda 的关系？

现在如何处理 Variable 和 numpy， tensor 的关系？

#### tensor operator

##### Create

`````Python
Tensor(sizes)    基础构造函数     ones(sizes)  全1 Tensor     zeros(*sizes)  全0Tensor
eye(*sizes)  对角线为1，不要求行列一致
rand(sizes)  均匀分布   randn(sizes) 标准分布
uniform(from,to) 均匀分布 normal(mean,std) 正态分布
arange(*sizes) 从s到e，步长为step
linspace(s,e,steps)  从s到e,均匀切分为steps份
randperm(m) 随机序列
t.tolist()/Tensor(l)    tensor和list的互换
t.sizes()/t.shape   返回t的形状
t.numel()   t中元素总个数
`````

##### Reshape

```Python
t.view(*sizes)  # reshape, -1 可以自动计算维度
t.unsquence(dim)  # 添加维度， 支持负数
t.squence(dim)  # 压缩dim维的"1"
t.resize()   t.resize_()
```

##### Slice  =>  reference cs231n

```Python
a[index]  # 第index 行
a[:,index]  # 第 index 列
a[row, column]  # row 行， cloumn 列
a[0, -1] #第零行， 最后一个元素
a[:index] # 前 index 行
a[:row, 0:1] # 前 row 行， 0和1列
a > 1 # return a ByteTensor 
a[a>1] # 选择 a > 1的元素， 等价于 a.masked_select(a>1)
```

##### 高级索引



##### Tensor Type



##### Element-wise

```Python
abs/sqrt/div/exp/fmod/log/pow...
cos/sin/asin/atan2/cosh...
ceil/round/floor/trunc
clamp(input, min, max)
sigmoid/tanh...
```

#####  归并操作

```Python
mean/sum/median/mode   均值/和/ 中位数/众数
norm/dist   范数/距离
std/var  标准差/方差
cumsum/cumprd  累加/累乘
```

#####  compare

```Python
gt >    lt <     ge >=     le <=   eq ==    ne != 
topk(input, k) -> (Tensor, LongTensor)
sort(input) -> (Tensor, LongTensor)
max/min => max(tensor)   max(tensor, dim)    max(tensor1, tensor2)
```

##### linear algebra

```python
trace  对角线元素之和(矩阵的迹)
diag  对角线元素
triu/tril  矩阵的上三角/下三角
mm/bmm   矩阵的乘法， batch的矩阵乘法
addmm/addbmm/addmv/addr/badbmm...  矩阵运算
t 转置
dor/cross 内积/外积
inverse 矩阵求逆
svd  奇异值分解
```

#### nn -> from torch import nn

TODO: PIL 读入是啥？ vs numpy vs tensor?

##### pad 填充

```Python
nn.ConstantPad2d(padding, value)
```

##### 卷积和反卷积

```Python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
```

##### 池化层

```
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
nn.AdaptiveMaxPool2d(output_size, return_indices=False)
nn.AdaptiveAvgPool2d(output_size)
nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
```

##### 全连接层

```
nn.Linear(in_features, out_features, bias=True)
```

##### 防止过拟合相关层

```python
nn.Dropout2d(p=0.5, inplace=False)
nn.AlphaDropout(p=0.5)
nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

##### 激活函数

```Python
TODO: add img？ 
nn.Softplus(beta=1, threshold=20)
nn.Tanh()
nn.ReLU(inplace=False)    
nn.ReLU6(inplace=False)
nn.LeakyReLU(negative_slope=0.01, inplace=False)
nn.PReLU(num_parameters=1, init=0.25)
nn.SELU(inplace=False)
nn.ELU(alpha=1.0, inplace=False)
```

##### RNN 

```Python
nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
nn.RNN(*args, **kwargs)
nn.LSTMCell(input_size, hidden_size, bias=True)
nn.LSTM(*args, **kwargs)
nn.GRUCell(input_size, hidden_size, bias=True)
nn.GRU(*args, **kwargs)
```

#####  Embedding

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None)
```

##### Sequential

```python
TODO: 如何使用Sequential？
nn.Sequential(*args)
```

##### loss functon

```python
nn.BCELoss(weight=None, size_average=True, reduce=True)
nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
nn.L1Loss(size_average=True, reduce=True)
nn.KLDivLoss(size_average=True, reduce=True)
nn.MSELoss(size_average=True, reduce=True)
nn.NLLLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
nn.NLLLoss2d(weight=None, size_average=True, ignore_index=-100, reduce=True)
nn.SmoothL1Loss(size_average=True, reduce=True)
nn.SoftMarginLoss(size_average=True, reduce=True)
nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-06, swap=False, size_average=True, reduce=True)
nn.CosineEmbeddingLoss(margin=0, size_average=True, reduce=True)
```

##### functional

```python
nn.functional，nn中的大多数layer，在functional中都有一个与之相对应的函数。nn.functional中的函数和nn.Module的主要区别在于，用nn.Module实现的layers是一个特殊的类，都是由class layer(nn.Module)定义，会自动提取可学习的参数。而nn.functional中的函数更像是纯函数，由def function(input)定义。
TODO: add init related function(什么时候用哪个?)
```

##### init

```Python
torch.nn.init.uniform
torch.nn.init.normal
torch.nn.init.kaiming_uniform
torch.nn.init.kaiming_normal
torch.nn.init.xavier_normal
torch.nn.init.xavier_uniform
torch.nn.init.sparse
```

#### optim -> form torch import optim

```python
optim.SGD(params, lr=<object object at 0x1113e69f0>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
optim.Optimizer(params, defaults)
```

#### save and load model

```python
torch.save(model.state_dict(), 'xxxx_params.pth')
model.load_state_dict(t.load('xxxx_params.pth'))
torch.save(model, 'xxxx.pth')
model.torch.load('xxxx.pth')
all_data = dict(
    optimizer = optimizer.state_dict(),
    model = model.state_dict(),
    info = u'model and optim parameter'
)
t.save(all_data, 'xxx.pth')
all_data = t.load('xxx.pth')
all_data.keys()
TODO: 导入导入 optimer 参数？ 作用是啥？
```

#### torchvision

##### models

```python
from torchvision import models
resnet34 = models.resnet34(pretrained=True, num_classes=1000)
```

##### data augmentation  -> from torchvision import transforms

```python
transforms.CenterCrop           transforms.Grayscale              transforms.ColorJitter          
transforms.Lambda               transforms.Compose                transforms.LinearTransformation 
transforms.FiveCrop             transforms.Normalize              transforms.functional           
transforms.Pad                  transforms.RandomAffine           transforms.RandomHorizontalFlip  
transforms.RandomApply          transforms.RandomOrder            transforms.RandomChoice         
transforms.RandomResizedCrop    transforms.RandomCrop             transforms.RandomRotation        
transforms.RandomGrayscale      transforms.RandomSizedCrop        transforms.RandomVerticalFlip   
transforms.ToTensor             transforms.Resize                 transforms.transforms                                                            
transforms.TenCrop              transforms.Scale                  transforms.ToPILImage
```

##### datasets

```Python
dataset = ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=1,shuffle=True)
for batch_datas, batch_labels in dataloader:
    ...
```

##### img process

```python
img = make_grid(next(dataiter)[0], 4) 
save_image(img, 'a.png')
```

#### Code Samples：

```Python
# torch.device object used throughout this script
device = torch.device("cuda" if use_cuda else "cpu")

model = MyRNN().to(device)

# train
total_loss = 0
for input, target in train_loader:
    input, target = input.to(device), target.to(device)
    hidden = input.new_zeros(*h_shape)  # has the same device & dtype as `input`
    ...  # get loss and optimize
    total_loss += loss.item()           # get Python number from 1-element Tensor

# evaluate
with torch.no_grad():                   # operations inside don't track history
    for input, target in test_loader:
        ...
```



