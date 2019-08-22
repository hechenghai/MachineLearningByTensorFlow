# 训练神经网络时batch大小的影响 #

----------


&ensp;&ensp;&ensp;&ensp;当训练一个神经网络模型时，会有许多超参数，这些超参数对模型的影响巨大，一旦超参数设置不好，就会让神经网络的效果还不如感知机。因此在面对神经网络这种容量很大的模型时，需要深刻理解其中超参数的意义及其对模型的影响。

&ensp;&ensp;&ensp;&ensp;首先我们看一下**epoch**、**batch**术语的解释。

&ensp;&ensp;&ensp;&ensp;**epoch** —— 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为epoch

&ensp;&ensp;&ensp;&ensp;**batch** —— 在不能讲数据一次性通过神经网络的时候，就需要将数据集分成几个batch

----------

## 基础知识回顾 ##
&ensp;&ensp;&ensp;&ensp;下图是神经网络一次迭代过程：
![](https://i.imgur.com/eiDzsFE.jpg)

&ensp;&ensp;&ensp;&ensp;由上图可以看出，首先选择n个样本组成一个batch，然后将batch直接丢进神经网络，得到输出结果，然后将输出的结果与样本正确的label计算loss损失，然后通过BP算法更新参数，这就是一次迭代的过程。
&ensp;&ensp;&ensp;&ensp;由此，最直观的超参数就是batch的大小————我们可以一次性将整个数据输入神经网络，让神经网络利用全部样本来计算迭代的梯度(即传统的梯度下降法)，也可以一次只输入一个样本(严格意义上的随机梯度下降法SGD)，也可以选择一个折中的方案，即每次只输入一部分样本，让其完成本轮迭代（即batch梯度下降法）。
&ensp;&ensp;&ensp;&ensp;一次性输入100个样本并迭代一次，跟一次性输入500个样本相比，主要区别有两种：
### 第一种 ####

总更新值 = 旧参数下更新值1+旧参数下更新值2+...+旧参数下更新值100；

新参数 = 旧参数 + 总更新值；

以二分类逻辑回归分类为例，loss损失为

<img src="https://latex.codecogs.com/gif.latex?J(\Theta)&space;=&space;-\frac{1}{m}\left&space;[\sum&space;y_{i}log&space;h_{\Theta}\left&space;(&space;x_{i}&space;\right&space;)&space;&plus;&space;(1-y_{i})log(1-h_{\Theta}\left&space;(&space;x_{i}&space;\right&space;))&space;\right&space;]" title="J(\Theta) = -\frac{1}{m}\left [\sum y_{i}log h_{\Theta}\left ( x_{i} \right ) + (1-y_{i})log(1-h_{\Theta}\left ( x_{i} \right )) \right ]" />

由该公式可知，标准的梯度下降法是计算所有样本的log损失和，参数只需要更新一次。  
### 第二种 ####                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                                                           
新参数1 = 旧参数 + 旧参数下计算更新值；

新参数2 = 新参数1 + 新参数1下计算更新值；

新参数3 = 新参数2 + 新参数2下计算更新值；

...

新参数100 =新参数100 + 新参数100下计算更新值；

还是以二分类逻辑回归分类为例，loss损失函数为：

<img src="https://latex.codecogs.com/gif.latex?J(\Theta)&space;=&space;-(log&space;h_{\Theta}\left&space;(&space;x_{i}&space;\right&space;)&space;&plus;&space;(1-y_{i})log(1-h_{\Theta}\left&space;(&space;x_{i}&space;\right&space;)))" title="J(\Theta) = -(log h_{\Theta}\left ( x_{i} \right ) + (1-y_{i})log(1-h_{\Theta}\left ( x_{i} \right )))" />

由该公式可知，随机梯度下降法是每次只计算一个样本的log损失，每次参数更新一次。

那么问题来了，哪一种方式更好呢？
### 收敛速度 #### 

----------

&ensp;&ensp;&ensp;&ensp;首先，我们分析哪种方法收敛更快。
&ensp;&ensp;&ensp;&ensp;我们假设每个样本相对于大自然真实分布的标准差为σ，那么根据概率统计的知识，很容易推出n个样本均值的标准差为<img src="https://latex.codecogs.com/gif.latex?\frac{\sigma&space;}{\sqrt{n}}" title="\frac{\sigma }{\sqrt{n}}" />
&ensp;&ensp;&ensp;&ensp;推导公式如下：
<img src="https://latex.codecogs.com/gif.latex?D\left&space;[&space;\bar{X}&space;\right&space;]&space;=&space;D\left&space;[&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;x_{i}&space;\right&space;]&space;=&space;\frac{1}{n^2}&space;\sum_{i=1}^{n}&space;D&space;\left[&space;x_{i}&space;\right]&space;=&space;\frac{1}{n^2}&space;\sum_{i=1}^{n}&space;\sigma&space;^2&space;=&space;\frac{1}{n^2}\cdot&space;n\sigma&space;^2&space;=&space;\frac{\sigma^2}{n}" title="D\left [ \bar{X} \right ] = D\left [ \frac{1}{n} \sum_{i=1}^{n} x_{i} \right ] = \frac{1}{n^2} \sum_{i=1}^{n} D \left[ x_{i} \right] = \frac{1}{n^2} \sum_{i=1}^{n} \sigma ^2 = \frac{1}{n^2}\cdot n\sigma ^2 = \frac{\sigma^2}{n}" />


&ensp;&ensp;&ensp;&ensp;从这里可以看出，使用样本来估计梯度的时候，1个样本带来σ的标准差，但是使用n个样本去估计梯度**并不能让样本均值的标准差线性降低**（也就是并不能让误差降低为原来的1/n，即无法达到σ/n），而n个样本的计算量确是线性的(每个样本都要平均的跑一边向前算法)。

&ensp;&ensp;&ensp;&ensp;由此可见，显然在同等的计算量之下，使用整个样本集合的收敛速度要远慢于使用少量样本的情况。使用整个样本集合，其确定的下降方向已经基本不再变化了。话句话说**要想收敛到同一个最优点，使用整个样本集合时，虽然迭代次数少，但是每次迭代的时间长，耗费的总时间是大于使用少量样本多次迭代的情况的。**

----------

&ensp;&ensp;&ensp;&ensp;那是不是样本越少，收敛速度越快？
&ensp;&ensp;&ensp;&ensp;理论上是这样的，使用单个单核CPU的情况下也确实是这样的。但是在实际工程中使用GPU训练时，跑一个样本花的时间与跑几十个样本甚至几百个样本的时间是一样的。因此实际工程中，从收敛速度的角度来说，小批量的样本集是最优的，也就是常说的*mini-batch*。在实际工程中*batch size*的大小一般是几十到几百不等。当然具体的大小还是跟显卡的计算能力有关的。

----------

&ensp;&ensp;&ensp;&ensp;但是是不是*batch size*越大越好呢？
&ensp;&ensp;&ensp;&ensp;我们知道，神经网络是个复杂的model，在实际中，神经网络的loss曲面往往是非凸的，这意味着可能有多个**局部最优点**，而且可能有**鞍点**
&ensp;&ensp;&ensp;&ensp;鞍点就是loss损失曲线中像马鞍一样形状的地方的中心点，改点也是一阶导数为0的点，但肯定不是最优点。如下图所示：

![](https://i.imgur.com/jt8Id9U.jpg)

&ensp;&ensp;&ensp;&ensp;想想一下，在鞍点处，横着看的话，鞍点就是一个极小值点，但是竖着看的话，鞍点就是极大值点。因此，鞍点容易给优化算法一个“我已经收敛了”的假象。但是实际上，**工程中却不怎么容易陷入很差劲的局部最优点或鞍点**。
&ensp;&ensp;&ensp;&ensp;因为，样本量少的时候回带来很大的方差，而这个大方差恰好会导致在梯度下降到很差的局部最优点和鞍点时不稳定，会因为一个大噪音而跳出局部最优点或者鞍点，从而有机会寻找耿优的最优点(此时学习率应该适当选择较小的)。与之相反，当样本量很多时，方差很小，对梯度的估计要准确和稳定的多，因此反而在差劲的局部最优点和鞍点时反而容易自信的呆着不走，从而导致神经网络收敛到很差的点上。因此，batch的size设置的不能太大也不能太小，因此实际工程中最常用的就是mini-batch，一般size设置为几十或者几百。对GPU而言，对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128.时，往往要比设置成10的倍数表现效果更佳。（未验证）

## 总结 ##

&ensp;&ensp;&ensp;&ensp;以上的讨论都是基于梯度下降法的，而且默认是一阶的情况(即没有利用二阶导数信息，仅仅使用一阶导数去优化)。因此对于SGD（随机梯度下降法）及其改良的一阶优化算法诸如Adagrad、Adam等是没有问题的，但是对于强大的二阶优化算法如共轭梯度法、L-BFGS来说，如果估计不好一阶导数，那么对二阶导数的估计会有更大的误差，这对于这些靠二阶导数吃饭的算法来说是致命的。

**调节****batch_size****对训练效果的影响如下图所示：**

![](https://i.imgur.com/mfsG2Ur.jpg)



+ batch_size太小，算法在200 epoches内不收敛。
+ 随着batch_size的增大，处理相同数量的速度越快。
+ 随着batch_size的增大，达到相同精度所需要的epoch数量越来越多。
+ 由于上述两种因素的矛盾，batch_size增大到某个时候，达到时间上的最优。
+ 由于最终收敛精度会陷入不同的局部极值，因此batch_size增大到某些时候，达到最终收敛精度上的最优。