# 纸质版勘误表

如何使用勘误？首先找到你的书的印次，接下来对着下表索引印次，该印次及其之后印次的勘误都是你书中所要注意的勘误，该印次前的所有勘误在当前印次均已修正。

## 第2版第6次印刷（2024.01）
- 127页，“式(8.7)的推导”中的最后一句话“两边同除$\frac{1}{2}$”改为“两边同乘$\frac{1}{2}$”（感谢@Acumen7）
- 181页，“式(10.14)的推导”中将式(10.14)化简成式(10.15)目标函数形式时，第⑥步中的$-\boldsymbol{x}_i^{\top}\mathbf{W}^{\top} \mathbf{x}_i$改为$-\boldsymbol{x}_i^{\top} \mathbf{W}\mathbf{W}^{\top} \mathbf{x}_i$（感谢@huishengye）
- 195页，“式(10.31)的目标函数形式”中的$\operatorname{tr}\left(\mathbf{Z}\mathbf{M}\mathbf{Z}\mathbf{Z}^{\top}\right)$改为$\operatorname{tr}\left(\mathbf{Z}\mathbf{M}\mathbf{Z}^{\top}\right)$（感谢@CoderKingXY）
- 86页，“样本内积$\boldsymbol{x}_i^{\mathrm{T}\boldsymbol{x}_j}$”改为“样本内积$\boldsymbol{x}_i^{\mathrm{T}}\boldsymbol{x}_j$”（感谢南瓜书读者交流群13的@.）
- 97页、98页，将其中的“$\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}$”改为“$\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{w}^{\phi} \boldsymbol{w}$”（感谢南瓜书读者交流群6的@Sodas）

## 第2版第5次印刷（2023.11）
- 98页，“6.6.5 核对率回归”中的第2个公式，其中的$\boldsymbol{x}_{i}$改为$\boldsymbol{z}_{i}$（感谢南瓜书读者交流群11的@[太阳]🌈）
- 13页，“2.3.6 式(2.12)~式(2.17)的解释”中的最后一段，将“式(2.17)的$\text{macro-}F1$是将$\text{macro-}P$和$\text{macro-}R$代入式(2.10)所得”改为“式(2.17)的$\text{micro-}F1$是将$\text{micro-}P$和$\text{micro-}R$代入式(2.10)所得”
- 46页，“3.4.1 式(3.32)的推导”中的第一段第一行，将“左下角”改为“右下角”
- 52页，“3.6 类别不平衡问题”的开头第一句话，将“对于类别平衡问题”改为“对于类别不平衡问题”

## 第2版第4次印刷（2023.10）
- 172页，$\|\mathbf{A}\|_F=\sum_{i=1}^m \sum_{j=1}^n\left|a_{i j}\right|^2$ 改为 $\|\mathbf{A}\|_F^{2}=\sum_{i=1}^m \sum_{j=1}^n\left|a_{i j}\right|^2$ （感谢@吴津宇）

## 第1版第12次印刷（2022.06）
- 式（3.9）中$\hat{\boldsymbol{x}}_i=(x_{1};...;x_{d};1)\in\mathbb{R}^{(d+1)\times 1}$改为$\hat{\boldsymbol{x}}_i=(x_{i1};...;x_{id};1)\in\mathbb{R}^{(d+1)\times 1}$（感谢@Link2Truth）

## 第1版第10次印刷（2021.12）
- 式（10.2）解释的最后一行，最后一个式子因为$1 + P^2\left(c^{*} | \boldsymbol{x}\right)\leqslant 2$改为$1 + P\left(c^{*} | \boldsymbol{x}\right)\leqslant 2$

## 第1版第7次印刷（2021.10）
- 92页，式(10.28)，“$n$行1列的单位向量”改为“$n$行1列的元素值全为1的向量”
- 95页，式(11.6)，“...降低因$w$的分量过太而导致...”改为“...降低因$w$的分量过大而导致...”（感谢@李伟豪work hard)
- 式(11.18)，求和可得下面的公式中第一行关于$\boldsymbol{b}$的列向量有笔误，最新表述参见：https://datawhalechina.github.io/pumpkin-book/#/chapter11/chapter11?id=_1118 （感谢@李伟豪work hard)

## 第1版第6次印刷（2021.07）
- 17页，式(3.37)，最后解析$\lambda$的取值那部分不太严谨，最新表述参见：https://datawhalechina.github.io/pumpkin-book/#/chapter3/chapter3?id=_337

## 第1版第4次印刷（2021.05）
- 17页，式(3.37)，解析的倒数第二行“将其代入$\mathbf{S}_{b} \boldsymbol{w}=\lambda \mathbf{S}_{b} \boldsymbol{w}$”改为“将其代入$\mathbf{S}_{b} \boldsymbol{w}=\lambda \mathbf{S}_{w} \boldsymbol{w}$”
- 80页，式(9.34)，$\mu$ 都改为粗体$\boldsymbol{\mu}$，表示向量 (感谢交流3群@橙子)
- 117页倒数第二行，式(12.42)，解析中“$\Phi(Z)$ 表示经验误差和泛化误差的上确界”改为“$\Phi(Z)$表示泛化误差和经验误差的差的上确界” (感谢交流3群@橙子)
- 145页，式（14.36），最后”即式（14.36）右侧的积分部分“上面的公式第二行$\Sigma_{z\ne j}$改为$\Sigma_{k\ne j}$ (感谢交流3群@橙子)
