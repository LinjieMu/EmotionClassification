## 1. 理论推导

设感知机的输入为![](https://latex.codecogs.com/svg.latex?n\times%20m)的特征集合![](https://latex.codecogs.com/svg.latex?\mathbf{A})，输出为![](https://latex.codecogs.com/svg.latex?n\times%201)的预测结果![](https://latex.codecogs.com/svg.latex?\mathbf{y})。

对于单个点，有：

![](https://latex.codecogs.com/svg.latex?y_i=\rm{sign}(\mathbf{w}^T\mathbf{x_i}+b))

为了保证损失函数的连续性，我没在此使用松弛的损失函数。其损失函数为：

![](https://latex.codecogs.com/svg.latex?L_i(\mathbf{w},b)=\hat%20y_i(\mathbf{w}^T\mathbf{x_i}+b))

若这个点的损失函数大于0，显然该点已经被正确分类，我们什么都不做。如果该点的损失函数小于0，显然该点已经被错误分类，我们需要增加损失函数的值，使之可以被正确分类。

损失函数的导数为：

![temp](https://tva1.sinaimg.cn/large/e6c9d24ely1h19kwk7zcqj202q02a0sj.jpg)

该损失函数的值越大说明准确率越高，因此优化过程中沿着梯度方向增加。

## 2. 过程实现

首先进行初始化。

```python
def __init__(self, x, y, w=None, b=None):
      num_sample, num_dims = x.shape
      np.random.seed(0)
      w = np.random.randn(num_dims, 1)/255/num_dims
      b = np.random.randn(1,1)
      self.x, self.y, self.w, self.b = x, y, w, b
      self.lr = 0.1
```

之后进行迭代，依次考虑每一个点，若该点的损失函数大于0，则分类正确，不予考虑，若小于0，则考虑该点的损失函数

```python
def update(self):
      gra_w = np.zeros((1200,1))
      gra_b = 0
      for p in range(len(self.x)):
          px = self.x[p, :]
          px = np.row_stack(px)
          py = self.y[p]
          Loss = py*(px.T.dot(self.w))
          # 如果Loss大于0，则什么都不做
          if Loss > 0:
              continue
          else:
              gra_w += py*px
              gra_b += py
      self.w += self.lr*gra_w
      self.b += self.lr*gra_b
```

预测方法：

```python
def predict(self):
      preds = self.x.dot(self.w)+self.b
      y_hat = np.sign(preds)
      return preds, y_hat.astype(int)
```

## 3. 结果展示

在学习率为0.1，迭代次数为100次的情况下：

初始情况：

![output4_1](https://tva1.sinaimg.cn/large/e6c9d24ely1h19ild9rnsj20ve02iq3q.jpg)

迭代10次后：

![output_4_10](https://tva1.sinaimg.cn/large/e6c9d24ely1h19ilqedvvj20ve02iwfa.jpg)

迭代50次后：

![output_4_50](https://tva1.sinaimg.cn/large/e6c9d24ely1h19ile8rdqj20ve02iwfa.jpg)

迭代100次后：

![output_4_100](https://tva1.sinaimg.cn/large/e6c9d24ely1h19ilfrqt0j20ve02iwfa.jpg)