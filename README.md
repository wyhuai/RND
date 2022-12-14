# GAN Prior based Null-Space Learning for Consistent Super-Resolution [AAAI 2023] 
## 📖[**Paper**](https://arxiv.org/abs/2211.13524)

[Yinhuai Wang](https://wyhuai.github.io/info/), Yujie Hu, [Jiwen Yu](https://scholar.google.com.hk/citations?user=uoRPLHIAAAAJ), [Jian Zhang](https://jianzhang.tech/)  
Peking University and PCL  

![image](https://user-images.githubusercontent.com/95485229/203699143-7d73ab43-7e40-4cdb-9cd1-d4293181f223.png)
The range-null space decomposition can improve the data consistency significantly. 

## How To Use Range-Null Space Decomposition In Your Image Restoration Tasks?
Since our method is independent to the choice of networks, it is queit **simple to apply our method to your own networks**😊. 

### Range-Null Space Decomposition (RND) for SR🔥
Let's take image SR task for example. Given an input LR image $\mathbf{y}$, one may uses a network $\mathcal{D}(\cdot)$ to yield the SR result, i.e.,

> $\mathbf{x}_{r}=\mathcal{D}(\mathbf{y})$

However, this SR result $\mathbf{x}_{r}$ may not assures low-frequency consistency.

We provide a very simple way to assure low-frequency consistency. We first define an average-pooling downsampler $\mathbf{A}$ and its pseudo-inverse $\mathbf{A}^{\dagger}$, then apply

> $\mathbf{x}_{r}=\mathcal{D}(\mathbf{y})$
> 
> $\hat{\mathbf{x}}=\mathbf{A}^{\dagger}\mathbf{y}+(\mathbf{I}-\mathbf{A}^{\dagger}\mathbf{A})\mathbf{x}\_{r}$

The result $\hat{\mathbf{x}}$ assures low-frequency consistency, i.e., $\mathbf{A}\hat{\mathbf{x}}\equiv\mathbf{y}$. Your may refer to our paper for detailed derivation.

Detailed implementation is here, you may copy these codes and apply to your own sr_backbone:

```python
import torch

def MeanUpsample(vec,scale):
    vec = vec.reshape(vec.size(0), 3, 1024//scale, 1024//scale)
    vec = vec.repeat(1, 1, scale, scale)
    vec = vec.reshape(vec.size(0), 3, scale, 1024//scale, scale, 1024//scale)
    out = vec.permute(0, 1, 3, 2, 5, 4).reshape(vec.size(0), 3, 1024, 1024)
    return out

class RND_sr(torch.nn.Module):
    def __init__(self, image_size=1024, sr_scale=8, sr_backbone):
        super(RND_sr, self).__init__()
        self.sr_backbone = sr_backbone()
        self.A = torch.nn.AdaptiveAvgPool2d((image_size//sr_scale,image_size//sr_scale))
        self.Ap = lambda z: MeanUpsample(z,sr_scale)
        
    def forward(self, x, y): 
        x = self.sr_backbone(y)
        x = self.Ap(y) + x - self.Ap(self.A(x))
        return x
```
You may try any sr_backbone. For the backbones used in this paper, you may refer to [GLEAN](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/glean/README.md) and [Panini](https://github.com/jianzhangcs/panini).

### RND for Image Colorization ⭐

```python
import torch

def color2gray(x):
    coef_r, coef_g, coef_b = 1/3, 1/3, 1/3
    x = x[:,0,:,:]*coef_r + x[:,1,:,:]*coef_g + x[:,2,:,:]*coef_b
    return x.repeat(1,3,1,1)

def gray2color(x):
    coef_r, coef_g, coef_b = 1/3, 1/3, 1/3
    x = x[:,0,:,:]
    base = coef_r**2 + coef_g**2 + coef_b**2
    return torch.stack((x*coef_r/base, x*coef_g/base, x*coef_b/base), 1)    

class RND_colorization(torch.nn.Module):
    def __init__(self, backbone):
        super(RND_colorization, self).__init__()
        self.backbone = backbone()
        self.A = lambda z: color2gray(z)
        self.Ap = lambda z: gray2color(z)
        
    def forward(self, x, y): 
        x = self.backbone(y)
        x = self.Ap(y) + x - self.Ap(self.A(x))
        return x
```

### RND for Compressed Sensing ⭐

```python
import torch

#initialize sampling matrix
C = channel
B = block_size
N = B * B
q = int(torch.ceil(th.Tensor([cs_ratio * N])))
H, W = image_size, image_size
U, S, V = torch.linalg.svd(th.randn(N, N))
A_weight = U.mm(V)  # CS sampling matrix weight
A_weight = A_weight.view(N, 1, B, B)[:q].to("cuda")

A = lambda z: torch.nn.functional.conv2d(z.view(batch_size * C, 1, H, W), A_weight, stride=B)
Ap = lambda z: torch.nn.functional.conv_transpose2d(z, A_weight, stride=B).view(batch_size, C, H, W)

class RND_cs(torch.nn.Module):
    def __init__(self, backbone):
        super(RND_cs, self).__init__()
        self.backbone = backbone()

    def forward(self, x, y): 
        x = self.backbone(y)
        x = Ap(y) + x - Ap(A(x))
        return x
```

## Discussion
When applied to small networks (e.g., ESRGAN) or complex dataset (e.g., DIV2K), RND may not bring significant improvement. Our method actually implies an assumption, that is, the network is "powerful" enough in generative performance. The "power" of the network is determined by the dataset and the network itself. For single-class datasets like the human faces, the data distribution is relatively easy to learn.

