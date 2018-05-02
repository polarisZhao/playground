# pytorch-playground
pytorch playground, include model

#### basic models

###### 基础模型系列

- [ ] vgg


- [x] vgg_bn

  ​


###### Inception 系列

- [x] googlenet
- [ ] Inception v2-v4
- [ ] Inception-resnet v1 - v2
- [ ] xception


###### resnet 系列

- [x] resnet

- [x] resnext

- [ ] wide-resnet = 5.1

- [x] senet

- [x] dpn =  5.1

- [x] densenet

  ​


###### 模型的压缩和加速系列

- [ ] mobilenet v1-v2  = 5.2
- [ ] shufflenet  = 5.2
- [ ] squeezenet = 5.2




5月3号-5 月 6号： check 一遍所有代码，再论文读一遍， 按照以下的tips 进行更新 

Tips： 

- 尽可能的兼容官方的代码[ paper release code， torchvision.models code ]


- 可读性，可重构性优先, 尽可能的可重构
- 后期要给出训练的 checkpoint 文件。这个后期在考虑，暂不考虑
- 给出详细的参考地址
- 招人帮忙检查最后的实现
- README 文件应该简介易懂
- 注意宣传
- 一致的地方：参数命名方式，文件头，调用的函数的接口，各种层的相关