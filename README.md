# playground

### MNIST_demo

### models

| Network                  |          top-1 error          |        top-5 error        |
| :----------------------- | :---------------------------: | :-----------------------: |
| AlexNet                  |             43.45             |           20.91           |
| VGG-11/13/16/19          |    30.98/30.07/28.41/27.62    |   11.37/10.75/9.62/9.12   |
| VGG-11/13/16/19 with BN  |    29.62/28.45/26.63/25.76    |   10.19/9.63/8.50/8.15    |
| ResNet-18/34/50/101/152  | 30.24/26.70/23.85/22.63/21.69 | 10.92/8.58/7.13/6.33/5.94 |
| SqueezeNet 1.0/1.1       |          41.81/25.35          |        19.38/7.83         |
| Densenet-121/169/201/161 |    25.35/24.00/22.80/22.35    |    7.83/7.00/6.43/6.20    |
| Inception v3             |             22.55             |           6.44            |
|                          |                               |                           |
|                          |                               |                           |

​


###### Inception 

- [ ] Inception v2-v4
- [ ] Inception-resnet v1 - v2
- [ ] xception


###### resnet 

- [x] resnext

- [x] senet

- [x] dpn =  5.1

  ​



###### model compression

- [ ] mobilenet v1-v2  = 5.2
- [ ] shufflenet  = 5.2










#### Detection

- [ ] Faster RCNN
- [ ] YOLO v3
- [ ] SSD



#### Segmentation

- [ ] FCN
- [ ] U-Net
- [ ] Mask Rcnn



#### GAN

- [ ] Origin GAN
- [x] DCGAN
- [ ] wGAN



5月3号-5 月 6号： check 一遍所有代码，再论文读一遍， 按照以下的tips 进行更新 

Tips： 


- 可读性，可重构性优先, 尽可能的可重构
- 后期要给出训练的 checkpoint 文件。这个后期在考虑，暂不考虑
- 给出详细的参考地址
- 招人帮忙检查最后的实现
- README 文件应该简介易懂
- 一致的地方：参数命名方式，文件头，调用的函数的接口，各种层的相关