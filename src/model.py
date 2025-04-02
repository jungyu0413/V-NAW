import pickle
from src.resnet import *
from src.resnet18 import *



#### resnet 18 ####
        
class NLA_r18(nn.Module):

    def __init__(self, args):
        super(NLA_r18, self).__init__()
        Resnet18 = resnet18()
        cp = torch.load('/workspace/NLA/src/resnet18_msceleb.pth')
        Resnet18.load_state_dict(cp['state_dict'])
        self.embedding = args.feature_embedding
        self.num_classes = args.num_classes
        self.features = nn.Sequential(*list(Resnet18.children())[:-2])  
        self.features2 = nn.Sequential(*list(Resnet18.children())[-2:-1])  
        self.fc = nn.Linear(self.embedding, self.num_classes)  
        self.exp_name = args.exp_name
        
                    
    def forward(self, x):        
        x = self.features(x)
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        #### 1, 2048, 1, 1   
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)

        return output
    


#### resnet 50 ####
        
class NLA_r50(nn.Module):

    def __init__(self):#, args):
        super(NLA_r50, self).__init__()
        Resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        with open('/workspace/NLA/weights/resnet50_ft_weight.pkl', 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
            Resnet50.load_state_dict(weights)
        self.embedding = 2048 # args.feature_embedding
        self.num_classes = 7 #args.num_classes
        
        self.features = nn.Sequential(*list(Resnet50.children())[:-2])  
        self.features2 = nn.Sequential(*list(Resnet50.children())[-2:-1])  
        self.fc = nn.Linear(self.embedding, self.num_classes)  

                    
    def forward(self, x):        
        x = self.features(x)
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        #### 1, 2048, 1, 1   
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)
            
        return output

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = NLA_r50()
        cp = torch.load('/workspace/NLA/weights/r50_best_mean.pth')
        base_model.load_state_dict(cp)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # Remove FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)  # (batch * clip_length, 2048, H, W)
        x = self.global_pool(x)  # (batch * clip_length, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch * clip_length, 2048)
        return x

class ResNet50FeatureExtractor_static(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = NLA_r50()
        cp = torch.load('/workspace/NLA/weights/r50_best_mean.pth')
        base_model.load_state_dict(cp)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # Remove FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)  # (batch * clip_length, 2048, H, W)
        static = self.global_pool(x)  # (batch * clip_length, 2048, 1, 1)
        output = static.view(static.size(0), -1)  # Flatten: (batch * clip_length, 2048)
        return output


class SwinTransformerFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # SwinTransformer 모델 초기화
        original_model = SwinTransFER(swin_num_features=768, num_classes=7, cam=True)
        
        # Pretrained weights 로드
        cp = torch.load('/workspace/NLA/weights/swin_best_mean.pth', map_location='cpu')
        original_model.load_state_dict(cp)
        
        # Encoder 부분만 추출
        self.feature_extractor = nn.Sequential(*list(original_model.children())[:-3])  # Remove FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)  # (batch, seq_len, hidden_dim)
        return x



class SwinTransformerFeatureExtractor_extrabranch(nn.Module):
    def __init__(self):
        super().__init__()
        # SwinTransformer 모델 초기화
        original_model = SwinTransFER(swin_num_features=768, num_classes=7, cam=True)
        
        # Pretrained weights 로드
        cp = torch.load('/workspace/NLA/weights/swin_best_mean.pth', map_location='cpu')
        original_model.load_state_dict(cp)
        
        # Encoder 부분만 추출
        self.feature_extractor = nn.Sequential(*list(original_model.children())[:-3])  # Remove FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)  # (batch, seq_len, hidden_dim)
        return x


#### Swin Transformer ####

from timm.models.layers import trunc_normal_
from torch.autograd import Variable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint

class SwinTransFER(torch.nn.Module):

    def __init__(self, swin_num_features=768, num_classes=7, cam=True):
        super().__init__()
        self.encoder = SwinTransformer(num_classes=512, drop_path_rate=0.1, attn_drop_rate=0.1)
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(swin_num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(swin_num_features, num_classes)
        self.cam = cam


    def forward(self, x):

        x = self.encoder.forward_features(x)
        x = self.norm(x)  # B L C

        feature = self.avgpool(x.transpose(1, 2))  # B C 1
        feature = torch.flatten(feature, 1)
        output = self.head(feature)

        if self.cam:

            fc_weights = self.head.weight
            fc_weights = fc_weights.view(1, self.num_classes, 768, 1, 1)
            fc_weights = Variable(fc_weights, requires_grad = False)

            # attention
            B, L, C = x.shape
            feat = x.transpose(1, 2).view(B, 1, C, 7, 7) # N * 1 * C * H * W
            hm = feat * fc_weights
            hm = hm.sum(2) # N * self.num_labels * H * W
            return output, hm
        
        else:
            return output



class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=112, patch_size=2, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.feature = nn.Sequential(
            nn.Linear(in_features=self.num_features, out_features=self.num_features, bias=False),
            nn.BatchNorm1d(num_features=self.num_features, eps=2e-5),
            nn.Linear(in_features=self.num_features, out_features=num_classes, bias=False),
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5)
        )
        self.feature_resolution = (patches_resolution[0] // (2 ** (self.num_layers-1)), patches_resolution[1] // (2 ** (self.num_layers-1)))


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        i = 0
        for layer in self.layers:
            i += 1
            x = layer(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.feature(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
    



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
    
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
    
    
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
    

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        with torch.cuda.amp.autocast(True):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
    

    

class VideoFeatureModel_static(nn.Module):
    def __init__(self, inc, base_model, temporal_model):
        super().__init__()
        self.base_model = base_model  # ResNet50 기반 feature extractor
        self.temporal_model = temporal_model  # Temporal Transformer or RNN model
        self.classifier = nn.Linear(512, 8)
        self.static_classifier = nn.Linear(inc, 8)

    def forward(self, x):
        # Input: (batch, clip_length, 3, 224, 224)
        batch, clip_length, c, h, w = x.size()

        # Reshape for ResNet50: (batch * clip_length, 3, 224, 224)
        x = x.view(batch * clip_length, c, h, w)

        # Feature extraction using ResNet50: (batch * clip_length, feature_dim)
        static_features = self.base_model(x)
        static_output = self.static_classifier(static_features)
        # Reshape back to temporal format: (batch, clip_length, feature_dim)
        static_features_dim = static_features.size(-1)
        static_features = static_features.view(batch, clip_length, static_features_dim)

        # Temporal processing: (batch, clip_length, class_num)
        output = self.temporal_model(static_features)
        output = self.classifier(output)
        return output, static_output



class VideoFeatureModel_concat(nn.Module):
    def __init__(self, base_model, temporal_model):
        super().__init__()
        self.base_model = base_model  # ResNet50 기반 feature extractor
        self.temporal_model = temporal_model  # Temporal Transformer or RNN model
        self.classifier = nn.Linear(512, 8)

    def forward(self, x, is_concat=False):
        """
        x: (batch, clip_length, 3, 224, 224) or (batch, clip_length*2, 3, 224, 224) if concat
        is_concat: 학습 중 concat된 입력인지 여부
        """
        batch, clip_length, c, h, w = x.size()
        half_clip = clip_length // 2  # clip_length를 절반으로 나누기

        # Reshape for ResNet50: (batch * clip_length, 3, 224, 224)
        x = x.view(batch * clip_length, c, h, w)

        # Feature extraction using ResNet50: (batch * clip_length, feature_dim)
        static_features = self.base_model(x)
        static_features_dim = static_features.size(-1)

        # Reshape back to temporal format
        static_features = static_features.view(batch, clip_length, static_features_dim)

        # 🔥 Training 시 `clip_length`를 절반씩 나눠야 함
        if is_concat:
            static_features_1 = static_features[:, :half_clip, :]  # 첫 절반 프레임
            static_features_2 = static_features[:, half_clip:, :]  # 다음 절반 프레임
            
            # Temporal processing separately
            output_1 = self.temporal_model(static_features_1)
            output_2 = self.temporal_model(static_features_2)
            
            # Classifier 적용
            output_1 = self.classifier(output_1)
            output_2 = self.classifier(output_2)

            return output_1, output_2

        else:
            # 🔥 Inference 시에는 그냥 한 번만 실행
            output = self.temporal_model(static_features)
            output = self.classifier(output)
            return output
        


class VideoFeatureModel_static(nn.Module):
    def __init__(self, base_model, temporal_model):
        super().__init__()
        self.base_model = base_model  # ResNet50 기반 feature extractor
        self.temporal_model = temporal_model  # Temporal Transformer or RNN model
        self.classifier = nn.Linear(512, 8)
        self.static_classifier = nn.Linear(512, 8)

    def forward(self, x, is_concat=False):
        """
        x: (batch, clip_length, 3, 224, 224) or (batch, clip_length*2, 3, 224, 224) if concat
        is_concat: 학습 중 concat된 입력인지 여부
        """
        batch, clip_length, c, h, w = x.size()
        half_clip = clip_length // 2  # clip_length를 절반으로 나누기

        # Reshape for ResNet50: (batch * clip_length, 3, 224, 224)
        x = x.view(batch * clip_length, c, h, w)

        # Feature extraction using ResNet50: (batch * clip_length, feature_dim)
        static_features = self.base_model(x)
        static_features_dim = static_features.size(-1)

        # Reshape back to temporal format
        static_features = static_features.view(batch, clip_length, static_features_dim)

        # 🔥 Training 시 `clip_length`를 절반씩 나눠야 함
        if is_concat:
            static_features_1 = static_features[:, :half_clip, :]  # 첫 절반 프레임
            static_features_2 = static_features[:, half_clip:, :]  # 다음 절반 프레임
            
            # Temporal processing separately
            output_1 = self.temporal_model(static_features_1)
            output_2 = self.temporal_model(static_features_2)
            
            # Classifier 적용
            output_1 = self.classifier(output_1)
            output_2 = self.classifier(output_2)

            static_output1 = self.static_classifier(static_features_1)
            static_output2 = self.static_classifier(static_features_2)

            return output_1, output_2, static_output1, static_output2

        else:
            # 🔥 Inference 시에는 그냥 한 번만 실행
            output = self.temporal_model(static_features)
            output = self.classifier(output)
            return output
        


class VideoFeatureModel_concat_va(nn.Module):
    def __init__(self, base_model, temporal_model):
        super().__init__()
        self.base_model = base_model  # ResNet50 기반 feature extractor
        self.temporal_model = temporal_model  # Temporal Transformer or RNN model
        self.val_estimator = nn.Linear(512, 1)
        self.aro_estimator = nn.Linear(512, 1)

    def forward(self, x, is_concat=False):
        """
        x: (batch, clip_length, 3, 224, 224) or (batch, clip_length*2, 3, 224, 224) if concat
        is_concat: 학습 중 concat된 입력인지 여부
        """
        batch, clip_length, c, h, w = x.size()
        half_clip = clip_length // 2  # clip_length를 절반으로 나누기

        # Reshape for ResNet50: (batch * clip_length, 3, 224, 224)
        x = x.view(batch * clip_length, c, h, w)

        # Feature extraction using ResNet50: (batch * clip_length, feature_dim)
        static_features = self.base_model(x)
        static_features_dim = static_features.size(-1)

        # Reshape back to temporal format
        static_features = static_features.view(batch, clip_length, static_features_dim)

        # 🔥 Training 시 `clip_length`를 절반씩 나눠야 함
        if is_concat:
            static_features_1 = static_features[:, :half_clip, :]  # 첫 절반 프레임
            static_features_2 = static_features[:, half_clip:, :]  # 다음 절반 프레임
            
            # Temporal processing separately
            output_1 = self.temporal_model(static_features_1)
            output_2 = self.temporal_model(static_features_2)
            
            # Classifier 적용
            val = self.val_estimator(output_1)
            aro = self.aro_estimator(output_1)

            return val, aro, output_1, output_2

        else:
            # 🔥 Inference 시에는 그냥 한 번만 실행
            output = self.temporal_model(static_features)
            val = self.val_estimator(output)
            aro = self.aro_estimator(output)

            return val, aro

class VideoFeatureModel_concat_au(nn.Module):
    def __init__(self, base_model, temporal_model):
        super().__init__()
        self.base_model = base_model  # ResNet50 기반 feature extractor
        self.temporal_model = temporal_model  # Temporal Transformer or RNN model
        self.au_detector = nn.Linear(512, 12)

    def forward(self, x, is_concat=False):
        """
        x: (batch, clip_length, 3, 224, 224) or (batch, clip_length*2, 3, 224, 224) if concat
        is_concat: 학습 중 concat된 입력인지 여부
        """
        batch, clip_length, c, h, w = x.size()
        half_clip = clip_length // 2  # clip_length를 절반으로 나누기

        # Reshape for ResNet50: (batch * clip_length, 3, 224, 224)
        x = x.view(batch * clip_length, c, h, w)

        # Feature extraction using ResNet50: (batch * clip_length, feature_dim)
        static_features = self.base_model(x)
        static_features_dim = static_features.size(-1)

        # Reshape back to temporal format
        static_features = static_features.view(batch, clip_length, static_features_dim)

        # 🔥 Training 시 `clip_length`를 절반씩 나눠야 함
        if is_concat:
            static_features_1 = static_features[:, :half_clip, :]  # 첫 절반 프레임
            static_features_2 = static_features[:, half_clip:, :]  # 다음 절반 프레임
            
            output_1 = self.temporal_model(static_features_1)
            output_2 = self.temporal_model(static_features_2)
            # Temporal processing separately
            output_1 = self.au_detector(output_1)
            output_2 = self.au_detector(output_2)
            

            return output_1, output_2

        else:
            # 🔥 Inference 시에는 그냥 한 번만 실행
            output = self.temporal_model(static_features)
            output = self.au_detector(output)

            return output


class TransEncoder(nn.Module):
    def __init__(self, inc=512, outc=512, dropout=0.6, nheads=1, nlayer=4):
        super(TransEncoder, self).__init__()
        self.nhead = nheads
        self.d_model = outc
        self.dim_feedforward = outc
        self.dropout = dropout
        self.conv1 = nn.Conv1d(inc, self.d_model, kernel_size=1, stride=1, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out = out.permute(2, 0, 1)
        out = self.transformer_encoder(out)
        out = out.permute(1, 0, 2)
        return out
    











import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        net = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(net + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,max_length=300, attention=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        if attention == True:
            layers += [AttentionBlock(max_length, max_length, max_length)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class AttentionBlock(nn.Module):
  """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """
  def __init__(self, dims, k_size, v_size, seq_len=None):
    super(AttentionBlock, self).__init__()
    self.key_layer = nn.Linear(dims, k_size)
    self.query_layer = nn.Linear(dims, k_size)
    self.value_layer = nn.Linear(dims, v_size)
    self.sqrt_k = math.sqrt(k_size)

  def forward(self, minibatch):
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys.transpose(2,1))
    # Use numpy triu because you can't do 3D triu with PyTorch
    # TODO: using float32 here might break for non FloatTensor inputs.
    # Should update this later to use numpy/PyTorch types of the input.
    mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
    mask = torch.from_numpy(mask).cuda()
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    logits.data.masked_fill_(mask, float('-inf'))
    probs = F.softmax(logits, dim=1) / self.sqrt_k
    read = torch.bmm(probs, values)
    return minibatch + read