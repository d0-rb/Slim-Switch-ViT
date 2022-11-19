import torch
import torch.nn as nn
from torch.nn import init, ParameterList
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numbers
from typing import List, Tuple
from torch import Tensor, Size


_shape_t = int | List[int] | Size


# Version of nn.Sequential which can accept multiple inputs/outputs between modules
# https://github.com/pytorch/pytorch/issues/19808#issuecomment-487291323
class MultipleSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class SwitchableLayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 switchable_buckets: int = 1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SwitchableLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.dims = tuple(i - len(normalized_shape) for i in range(len(normalized_shape)))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.switchable_buckets = switchable_buckets
        if self.elementwise_affine:
            self.weights = Parameter(torch.empty((self.switchable_buckets, *self.normalized_shape), **factory_kwargs))  # First dim is bucket dim
            self.biases = Parameter(torch.empty((self.switchable_buckets, *self.normalized_shape), **factory_kwargs))
            # self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            # self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.buckets = Parameter(torch.empty(self.switchable_buckets, 2, **factory_kwargs), requires_grad=False)  # Last dim is for mean, var
        self.bucket_amounts = Parameter(torch.empty(self.switchable_buckets, device=device, dtype=torch.long), requires_grad=False)  # Number of samples that have been sent to each bucket

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            # init.ones_(self.weight)
            # init.zeros_(self.bias)
            init.ones_(self.weights)
            init.zeros_(self.biases)

    # Passing in bucket means we are in training curriculum stage, not passing bucket means we want it to be switched
    # Input is (*, normalized_shape[0], normalized_shape[1], ...)
    def forward(self, input: Tensor, bucket: Tensor | int | None = None) -> Tensor:
        # return F.layer_norm(
        #     input, self.normalized_shape, self.weights[0], self.biases[0], self.eps), bucket

        unnormalized_dims = input.shape[:self.dims[0]]
        
        if isinstance(bucket, int):
            assert 0 <= bucket < self.switchable_buckets, "Passed bucket index for updating bucket statistics is invalid!"
            bucket = torch.tensor(bucket)
        if isinstance(bucket, Tensor):  # Should be shape (*)
            assert 0 <= bucket.min() and bucket.max() < self.switchable_buckets, "Passed bucket tensor for updating bucket statistics has invalid indices!"
            assert not torch.is_floating_point(bucket), "Passed bucket tensor should be dtype int (or long or other equivalent)!"
            bucket = bucket.broadcast_to(unnormalized_dims)  # (*)

        mean = torch.mean(input, dim=self.dims, keepdim=True)  # (*, 1, 1, ...)
        input_mean_diff = input - mean
        var = torch.square(input_mean_diff).mean(dim=self.dims, keepdim=True)  # (*, 1, 1, ...)

        normalized = input_mean_diff / torch.sqrt(var + self.eps)  # Same shape as input

        if bucket is None:  # If we want a bucket to be selected
            mean_var_combined = torch.stack([mean.squeeze(), var.squeeze()], dim=-1)  # (*, 2)

            bucket_distances = torch.cdist(mean_var_combined, self.buckets, p=2)  # (*, self.switchable_buckets)
            selected_buckets = bucket_distances.argmin(dim=-1)  # (*) Gets indices of lowest distances from each bucket

            # selected_buckets = torch.cdist(torch.stack([mean.squeeze(), var.squeeze()], dim=-1), self.buckets, p=2).argmin(dim=-1)  # (*) Gets indices of lowest distances from each bucket

            # Will separate normalized by bucket
            # normalized_separate = normalized.unsqueeze(0).expand(self.switchable_buckets, *normalized.shape)  # (# of buckets, *, *normalized_shape)
            # indices = torch.arange(end=self.switchable_buckets).view(-1, *((1,) * len(normalized.shape)))  # (# of buckets, 1, 1, ...)
            # transformed_normalized_separate = normalized_separate * self.weights.view(self.switchable_buckets, *((1,) * len(unnormalized_dims)), *self.normalized_shape) + self.biases.view(self.switchable_buckets, *((1,) * len(unnormalized_dims)), *self.normalized_shape)
            # transformed_normalized_separate = transformed_normalized_separate * (bucket == indices)  # Same shape, but now each element along dim 0 holds normalized values for a particular bucket            
            # transformed_normalized = transformed_normalized_separate.sum(dim=0)
            
            # transformed_normalized = normalized * self.weights[selected_buckets] + self.biases[selected_buckets]
            
            # transformed_normalized = normalized * self.weight + self.bias

            transformed_normalized = normalized
            
            if self.elementwise_affine:
                for i in range(self.switchable_buckets):
                    transformed_normalized[bucket == i] = transformed_normalized[bucket == i] * self.weights[i] + self.biases[i]
                    # normalized[bucket == i] = normalized[bucket == i] * self.weights[i] + self.biases[i]

            return transformed_normalized, selected_buckets
            # return normalized, selected_buckets
        else:  # If we want to update a bucket's statistics by sending this batch to the passed bucket(s)
            added_amts = bucket.flatten().bincount()
            self.bucket_amounts += added_amts

            nonzero = self.bucket_amounts != 0
            # Relative added amounts; e.g., an input with 20 samples sent to a bucket with 80 contributed 20% to the new total
            added_amts_rel = torch.zeros_like(added_amts)
            added_amts_rel[nonzero] = added_amts[nonzero] / self.bucket_amounts[nonzero]

            # Will hold mean of means for each bucket
            means = mean.unsqueeze(0).expand(*added_amts.shape, *mean.shape).view(*added_amts.shape, -1)  # (# of given buckets, 1)
            indices = torch.arange(end=added_amts.size(dim=0)).view(-1, 1)
            means = means * (bucket == indices)  # Same shape, but now each element along dim 0 holds means for a particular bucket
            means = means.mean(1)  # Take mean of means for each bucket

            # Will hold mean of vars for each bucket
            vars = var.unsqueeze(0).expand(*added_amts.shape, *var.shape).view(*added_amts.shape, -1)  # (# of given buckets, 1)
            vars = vars * (bucket == indices)  # Same shape, but now each element along dim 0 holds vars for a particular bucket
            vars = vars.mean(1)  # Take mean of vars for each bucket

            # Updating means
            self.buckets[..., 0][nonzero] = (means * added_amts_rel + self.buckets[..., 0] * (1 - added_amts_rel))[nonzero]
            # Updating vars
            self.buckets[..., 1][nonzero] = (vars * added_amts_rel + self.buckets[..., 1] * (1 - added_amts_rel))[nonzero]

            transformed_normalized = normalized * self.weights[bucket] + self.biases[bucket]

            return transformed_normalized, bucket


    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.dims = normalized_shape if isinstance(normalized_shape, int) else tuple(i - len(normalized_shape) for i in range(len(normalized_shape)))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        mean = torch.mean(input, dim=self.dims, keepdim=True)  # (*, 1, 1, ...)
        input_mean_diff = input - mean
        var = torch.square(input_mean_diff).mean(dim=self.dims, keepdim=True)  # (*, 1, 1, ...)

        normalized = input_mean_diff / torch.sqrt(var + self.eps)  # Same shape as input

        transformed_normalized = normalized * self.weight + self.bias if self.elementwise_affine else normalized  # Same shape as input
        
        return transformed_normalized

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=197):
        super().__init__()
        self.num_heads = num_heads
        self.num_patches = num_patches
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

SlimAttention = Attention

class SparseAttention(Attention):
    def __init__(self, attn_module, head_search=False, uniform_search=False):
        super().__init__(attn_module.qkv.in_features, attn_module.num_heads, True, attn_module.scale, attn_module.attn_drop.p, attn_module.proj_drop.p)
        self.is_searched = False
        self.num_gates = attn_module.qkv.in_features // self.num_heads
        if head_search:
            self.zeta = nn.Parameter(torch.ones(1, 1, self.num_heads, 1, 1))
        elif uniform_search:
            self.zeta = nn.Parameter(torch.ones(1, 1, 1, 1, self.num_gates))
        else:
            self.zeta = nn.Parameter(torch.ones(1, 1, self.num_heads, 1, self.num_gates))
        self.searched_zeta = torch.ones_like(self.zeta)
        self.patch_zeta = nn.Parameter(torch.ones(1, self.num_patches, 1)*3)
        self.searched_patch_zeta = torch.ones_like(self.patch_zeta)
        self.patch_activation = nn.Tanh()
    
    def forward(self, x):
        z_patch = self.searched_patch_zeta if self.is_searched else self.patch_activation(self.patch_zeta)
        x *= z_patch
        B, N, C = x.shape
        z = self.searched_zeta if self.is_searched else self.zeta
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, H, N, d(C/H)
        qkv *= z
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B, H, N, d

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_zeta(self):
        return self.zeta, self.patch_activation(self.patch_zeta)
    
    def compress(self, threshold_attn):
        self.is_searched = True
        self.searched_zeta = (self.zeta>=threshold_attn).float()
        self.zeta.requires_grad = False
        
    def compress_patch(self, threshold_patch=None, zetas=None):
        self.is_searched = True
        zetas = torch.from_numpy(zetas).reshape_as(self.patch_zeta)
        self.searched_patch_zeta = (zetas).float().to(self.zeta.device)
        self.patch_zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True
        self.patch_zeta.requires_grad = True

    def get_params_count(self):
        dim = self.qkv.in_features
        active = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1:
            active*=self.num_gates
        elif self.zeta.shape[2] == 1:
            active*=self.num_heads
        total_params = dim*dim*3 + dim*3
        total_params += dim*dim + dim
        active_params = dim*active*3 + active*3
        active_params += active*dim +dim
        return total_params, active_params
    
    def get_flops(self, num_patches, active_patches):
        H = self.num_heads
        N = num_patches
        n = active_patches
        d = self.num_gates
        sd = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1: # Head Elimination
            sd*=self.num_gates
        elif self.zeta.shape[2] == 1: # Uniform Search
            sd*=self.num_heads
        total_flops = N * (H*d * (3*H*d)) + 3*N*H*d #linear: qkv
        total_flops += H*N*d*N + H*N*N #q@k
        total_flops += 5*H*N*N #softmax
        total_flops += H*N*N*d #attn@v
        total_flops += N * (H*d * (H*d)) + N*H*d #linear: proj
        
        active_flops = n * (H*d * (3*sd)) + 3*n*sd #linear: qkv
        active_flops += n*n*sd + H*n*n #q@k
        active_flops += 5*H*n*n #softmax
        active_flops += n*n*sd #attn@v
        active_flops += n * (sd * (H*d)) + n*H*d #linear: proj
        return total_flops, active_flops

    @staticmethod
    def from_attn(attn_module, head_search=False, uniform_search=False):
        attn_module = SparseAttention(attn_module, head_search, uniform_search)
        return attn_module

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

SlimMlp = Mlp

class SparseMlp(Mlp):
    def __init__(self, mlp_module):
        super().__init__(mlp_module.fc1.in_features, mlp_module.fc1.out_features, mlp_module.fc2.out_features, act_layer=nn.GELU, drop=mlp_module.drop.p)
        self.is_searched = False
        self.num_gates = mlp_module.fc1.out_features
        self.zeta = nn.Parameter(torch.ones(1, 1, self.num_gates))
        self.searched_zeta = torch.ones_like(self.zeta)  
    
    def forward(self, x, patch_zeta=None):
        if patch_zeta is not None:
            x*=patch_zeta
        z = self.searched_zeta if self.is_searched else self.get_zeta()
        x = self.fc1(x)
        x = self.act(x)
        x *= z # both fc1 and fc2 dimensions eliminated here
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def get_zeta(self):
        return self.zeta
    
    def compress(self, threshold):
        self.is_searched = True
        self.searched_zeta = (self.get_zeta()>=threshold).float()
        self.zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True

    def get_params_count(self):
        dim1 = self.fc1.in_features
        dim2 = self.fc1.out_features
        active_dim2 = self.searched_zeta.sum().data
        total_params = 2*(dim1*dim2) + dim1 + dim2
        active_params = 2*(dim1*active_dim2) + dim1 + active_dim2
        return total_params, active_params
    
    def get_flops(self, num_patches, active_patches):
        total_params, active_params = self.get_params_count()
        return total_params*num_patches, active_params*active_patches

    @staticmethod
    def from_mlp(mlp_module):
        mlp_module = SparseMlp(mlp_module)
        return mlp_module
