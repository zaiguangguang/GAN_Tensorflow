# GAN_Tensorflow

## 定义判别器


```python
def discriminator(x):
    # 计算D_h1=ReLU（x*D_W1+D_b1）,该层的输入为含784个元素的向量
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)

    # 计算第三层的输出结果。因为使用的是Sigmoid函数，则该输出结果是一个取值为[0,1]间的标量（见上述权重定义）
    # 即判别输入的图像到底是真（=1）还是假（=0）
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    # 返回判别为真的概率和第三层的输入值，输出D_logit是为了将其输入tf.nn.sigmoid_cross_entropy_with_logits()以构建损失函数
    return D_prob, D_logit
```
