# 机器学习笔记

## 正则化

$$
Loss + \lambda J(f)
$$
J(f) 是正则化项，一般与模型的复杂度成正比，模型越复杂，正则化值越大。

一般用参数向量的L1范数和L2范数。
$$
L1=\lambda || w||_{1}
$$

$$
L2=\frac{\lambda}{2}||w||^{2}
$$

L1就是向量中绝对值的和，L2就是平方和开方

##生成模型与判别模型

生成模型：由数据学习x与y的联合概率，然后再求出条件概率当作模型。
$$
P(Y|X)=\frac{P(X,Y)}{P(X)}
$$
比如HMM模型、朴素贝叶斯模型。好处就是收敛快？有隐变量也可以（无监督）。

判别模型：直接学习条件概率P(Y|X)，比如CRF模型、最大熵模型。

## 朴素贝叶斯模型

* 生成模型

  基于特征条件独立假设，学习输入输出的联合概率分布；然后在解码的时候，基于这个模型对给定输入x，利用贝叶斯定理求出后验概率最大的输出y。

* 怎么学习联合概率分布P(X,Y)?

  * $$
    P(X,Y)=P(Y)*P(X|Y)
    $$

  * 所以先学习P(Y),
    $$
    P(Y=c_{k})=\frac{\sum_{i=1}^N{I(y_{i}=c_{k})}}{N}
    $$

  * 然后再学习P(X|Y)
    $$
    P(X|Y)=P(x_{1},...,x_{n}|Y=c_{k}),k=1,2,...,K
    $$
    然而如果直接学习这个，就会有指数级别的参数数量。所以就需要条件独立假设。

  * 条件独立假设
    $$
    P(X=x|Y=c_{k})=P(x^{1},...,x^{n}|Y=c_{k})=\prod_{j=1}^nP(X^j=x^j|Y=c_{k})
    $$

  * 转而学习由条件独立拆开的
    $$
    P(x^j=a|Y=c_{k})=\frac{\sum_{i=1}^NI(x^j=a,y_{i}=c_{k})}{\sum_{i=1}^NI(y_{i}=c_{k})}
    $$

  * 最终

  $$
  y=f(x)=\arg \max _{c_{k}} \frac{P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}
  $$

  * 然后因为分母一样，所以去掉分母：
    $$
    y=f(x)=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
    $$

## 信息量、熵、相对熵、交叉熵
* 信息量
我们知道：当一个事件发生的概率为 
$$
p(x)
$$

​		那么这件事情的信息量就为：
$$
- log(p(x))
$$
​		概率越小信息量越大，”狗咬人、人咬狗“

* 熵

  信息量的期望就是熵，所以熵的公式为：

  假设 事件![[公式]](https://www.zhihu.com/equation?tex=X) 共有n种可能，发生 ![[公式]](https://www.zhihu.com/equation?tex=x_i++) 的概率为 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_i%29) ，那么该事件的熵 ![[公式]](https://www.zhihu.com/equation?tex=H%28X%29) 为：

  ![[公式]](https://www.zhihu.com/equation?tex=H%28X%29%3D%E2%88%92%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp%28x_i%29log%28p%28x_i%29%29%7D)

  然而有一类比较特殊的问题，比如投掷硬币只有两种可能，字朝上或花朝上。买彩票只有两种可能，中奖或不中奖。我们称之为0-1分布问题（二项分布的特例），对于这类问题，熵的计算方法可以简化为如下算式：

  ![[公式]](https://www.zhihu.com/equation?tex=H%28X%29%3D%E2%88%92%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp%28x_i%29log%28p%28x_i%29%29%7D+%3D-p%28x%29log%28p%28x%29%29+-+%281-p%28x%29%29log%281-p%28x%29%29)

  

* 相对熵

  相对熵又称KL散度,如果我们对于同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度（Kullback-Leibler (KL) divergence）来衡量这两个分布的差异。

  在机器学习中，P往往用来表示样本的真实分布，Q用来表示模型所预测的分布，那么KL散度就可以计算两个分布的差异，也就是Loss损失值。

  

  ![[公式]](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28p%7C%7Cq%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp%28x_i%29log%28%5Cfrac%7Bp%28x_i%29%7D%7Bq%28x_i%29%7D%29%7D)

* 交叉熵

  将KL散度公式进行变形：

  ![[公式]](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28p%7C%7Cq%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp%28x_i%29log%28%5Cfrac%7Bp%28x_i%29%7D%7Bq%28x_i%29%7D%29%7D+%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp%28x_i%29log%28p%28x_i%29%29%7D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp%28x_i%29log%28q%28x_i%29%29%7D+%3D-H%28p%28x%29%29+%2B+%5B-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp%28x_i%29log%28q%28x_i%29%29%7D%5D)

  等式的前一部分恰巧就是p的熵，等式的后一部分，就是交叉熵：

  ![[公式]](https://www.zhihu.com/equation?tex=H%28p%2C+q%29+%3D+-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp%28x_i%29log%28q%28x_i%29%29%7D)

  

  在机器学习中，我们需要评估label和predicts之间的差距，使用KL散度刚刚好，即 ![[公式]](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28y%7C%7C%5Ctilde%7By%7D%29) ，由于KL散度中的前一部分−H(y)不变，故在优化过程中，只需要关注交叉熵就可以了。所以一般在机器学习中直接用用交叉熵做loss，评估模型。
  
## BatchNorm and LayerNorm
* 首先为什么要Norm：据说可以让加快模型收敛速度，减少训练需要的时间；另外因为经过多重矩阵计算，数值往往会变大，Norm可以归一化，也缓解梯度爆炸之类的。
* BatchNorm
  对一个batch中的某一特征维度进行归一化，通常在NLP中不用。因为一个batch中的句子长度不一致，虽然可以pad成相同的长度，但是要和pad的特征进行归一化，这不合理。
* LayerNorm
  对一个instance的所有特征维度进行归一化，在NLP中常用，通常是[batch_size, seq_len, dim]中的dim进行归一化。
  
## Transformer的多头注意力机制
* 为什么要用多头注意力机制，比单个头好在哪里？
  关于这个问题其实还没有定论，说不清，有几点：
  * 把向量划分为多个子向量，然后对这多个子空间去做attention，让模型去关注不同的方面
  * 计算量和单个头是差不多的
  * 这样做有点像dropout，也有点像模型内部的集成
