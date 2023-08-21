from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from operator import itemgetter


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Removed to comply with DepthCLIP design. DepthCLIP should output B, HW, C, but using self.attnpool outputs Bx1xC.
        # x = self.attnpool(x)

        ## Below: replacement not present in original CLIP.
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_size = patch_size

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        pos_emb = F.interpolate(self.positional_embedding.permute(1, 0).unsqueeze(0), size=[x.shape[1]], mode='linear').reshape(-1, x.shape[1]).permute(1, 0)
        x = x + pos_emb.to(x.dtype)

        # x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x[:, 1:, :])  # B, HW, C

        if self.proj is not None:
            x = x @ self.proj

        # x = x.permute(0, 2, 1).unsqueeze(-1)
        # x = x.view(x.shape[0], x.shape[1], out_height, out_width)
        # x = x.contiguous()

        return x    # B, C, H, W (where H and W are patch indices, not pixel coords)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))


    def encode_text(self, text_indices, extra_tkns_reverse_lookup={}, extra_tkns_fwd_lookup={}, img_f_tkns=None):
        """
        Encode pre-tokenized text to give sentence embeddings.
        Args:
            :param text_indices (list of torch.IntTensors): List of int tensors containing integer token indices.
            :param extra_tkns_reverse_lookup (dict): dict of mappings from integer token indices to tuple of (source word, 512d token)
            :param extra_tkns_fwd_lookup (dict): dict of mappings from text token to integer token (e.g. "<|depth_0|>" to 49409)
            :param img_f_tkns (optional, torch.Tensor (BxHWx512)): A special override for use when token "<|imgf_0|>" is used.
                                                                   Each token in the BxHW dim will be substituted for "<|imgf_0|>" in turn.
        """
        # Do lookup in two goes: first for the regular tokens, then the extra tokens, then merge the two.
        x, extra_x_idx = self.idx_to_tkn_normal(text_indices)   # Get the normal (in-vocab) tokens, and indices of any out-of-vocab tokens.
        x, text_indices = self.handle_extra_tokens(x, text_indices, extra_x_idx, extra_tkns_reverse_lookup, extra_tkns_fwd_lookup, img_f_tkns)

        # Get prompt embeddings.
        # If there's too many dimensions (i.e. there's img_f_tkns) then break into chunks and do iteratively
        if len(x.shape) > 4:
            batch_embeddings_list = []
            dims = x.shape
            batches = torch.split(x.view(-1, dims[-2], dims[-1]), 400)
            batch_indices = torch.split(text_indices.view(-1, text_indices.shape[-1]), 400)
            for i, batch in enumerate(batches):
                batch_embeddings_list.append(self.encode_tokens(text_indices=batch_indices[i], tokens=batch))
            
            x = torch.cat(batch_embeddings_list, dim=0).view([*dims[0:3], batch_embeddings_list[0].shape[-1]])

        else:
            if len(x.shape) == 4:
                tmp = x.shape
                x = x.view(-1, tmp[2], tmp[3])
                x = self.encode_tokens(text_indices=text_indices, tokens=x)
                x = x.view(tmp[0], tmp[1], -1)
            else:
                x = self.encode_tokens(text_indices=text_indices, tokens=x)

        return x


    def idx_to_tkn_normal(self, text_indices):
        """ Originally part of self.encode_text(). Does lookup for all frozen (in-vocab) CLIP tokens at once.
        Returns:
            - x: the text embeddings
            - extra_x_idx: the indices of where an out-of-vocab token is found (these need to be replaced by something else.)
        """
        # Do lookup in two goes: first for the regular tokens, then the extra tokens, then merge the two.
        is_extra_tkn_mask = text_indices >= self.vocab_size
        x = self.token_embedding(text_indices * torch.logical_not(is_extra_tkn_mask))
        extra_x_idx = is_extra_tkn_mask.nonzero()
        return x, extra_x_idx


    def handle_extra_tokens(self, half_tokenized, text_indices, extra_tkn_idx, extra_tkns_reverse_lookup, extra_tkns_fwd_lookup, img_f_tkns=None):
        """Take the half-tokenized input (Nx77x512) and replace out-of-vocab tokens with the relevant extra tokens.
        Args:
            :param half_tokenized (torch.Tensor, Nx77x512): N sequences of 512d tokens of length 77. Tokens for out-of-vocab words will be wrong and will need to be corrected in this fn.
            :param text_indices (torch.IntTensor, Nx77): N sequences of the token indices for the vocab tokens. Out of vocab tokens are just larger than the vocab size.
            :param extra_x_idx (torch.IntTensor, Mx2): Location of the M different out-of-vocab tokens in the sequence. First element of each specifies sequence, second specifies token in sequence.
            :param extra_tkns_reverse_lookup (dict): Dict mapping from index token to tuple of (word token, learnable parameters). Won't work for img_f.
            :param extra_tkns_fwd_lookup (dict): Dict mapping from text token (e.g. "<|depth_0|>" to index token.)
            :param img_f_tkns (default None, torch.Tensor, BxHWx512): image features. If present, prompts will be replicated for each img_f_tkn.
                                                                      E.g. BHWNx77x512 prompts are compiled, then BHWNx1x512 embeddings should end up returned.
        """
        # oov_token_indices are integer out-of-vocabulary token indices (that need replacing with something else).

        # Done in three or two stages:
        #   0. (Only if img_f_tkns is not None): expand the number of prompts to accommodate img_f, from Nx77x512 -> BxHWxNx77x512
        #   1. Handle the "normal" (non img_f) token substitutions
        #   2. Handle the img_f token substitutions, if any need to be done. **There must be only one per prompt, with token "<|imgf_0|>"**.
        if img_f_tkns is not None:
            assert extra_tkns_fwd_lookup.get("<|imgf_0|>") is not None
            
            half_tokenized = half_tokenized.expand([img_f_tkns.shape[0], img_f_tkns.shape[1], -1, -1, -1]).clone() # Nx77x512 -> BxHWxNx77x512
            text_indices = text_indices.repeat(img_f_tkns.shape[0], img_f_tkns.shape[1], 1, 1) # Nx77 -> BxHWxNx77

            # Overwrite the input extra_tkn_idx because the one passed in includes imgf tokens, which will be handled separately.
            is_extra_tkn_mask = (text_indices >= self.vocab_size) & (text_indices != extra_tkns_fwd_lookup.get("<|imgf_0|>"))
            extra_tkn_idx = is_extra_tkn_mask.nonzero()

            is_imgf_mask = text_indices == extra_tkns_fwd_lookup.get("<|imgf_0|>")
            imgf_tkn_idx = is_imgf_mask.nonzero()
        
        if extra_tkns_fwd_lookup is not None:
            tmp_keys = extra_tkns_fwd_lookup.keys()
            if sum(["img" in k for k in tmp_keys]) < len(tmp_keys):
                    oov_token_indices = text_indices[extra_tkn_idx.split(1, dim=1)].squeeze(-1)
                    if isinstance(extra_tkns_reverse_lookup[oov_token_indices.tolist()[0]], torch.Tensor):
                        oov_replacements = torch.stack(itemgetter(*oov_token_indices.tolist())(extra_tkns_reverse_lookup), dim=1)
                        half_tokenized = half_tokenized.unsqueeze(0).repeat(oov_replacements.shape[0], 1, 1, 1)
                        half_tokenized[:, extra_tkn_idx[:, 0], extra_tkn_idx[:, 1], :] = oov_replacements
                        text_indices = text_indices.unsqueeze(0).repeat(half_tokenized.shape[0], 1, 1)
                    else:
                        _, oov_replacements = list(zip(*itemgetter(*oov_token_indices.tolist())(extra_tkns_reverse_lookup)))
                        oov_replacements = torch.cat(oov_replacements, dim=0)
                        half_tokenized[extra_tkn_idx.split(1, dim=1)] = oov_replacements.unsqueeze(1)
        
        if img_f_tkns is not None:
            # imgf_token_indices = text_indices[imgf_tkn_idx.split(1, dim=1)].squeeze(-1)
            img_f_tkns = img_f_tkns.unsqueeze(2).unsqueeze(2).expand(-1, -1, half_tokenized.shape[2], half_tokenized.shape[3], -1) # BxHWx512 -> BxHWxNx77x512
            imgf_replacements = img_f_tkns[imgf_tkn_idx.split(1, dim=1)].clone()
            half_tokenized[imgf_tkn_idx.split(1, dim=1)] = imgf_replacements

        return half_tokenized, text_indices


    def encode_tokens(self, text_indices, tokens):
        """ take batch of N tokens (BxNx512), add positional embedding, and run through transformer. """
        x = tokens
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Merge word tokens into sentence token

        x = x[torch.arange(x.shape[0]), (text_indices == (self.vocab_size - 1)).nonzero()[:, -1]] @ self.text_projection
        # x = x[torch.arange(x.shape[0]), 7] @ self.text_projection

        return x


    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
