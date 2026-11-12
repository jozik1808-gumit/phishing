import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, MarkupLMModel
from common import BERT, MARKUPLM, MARKUP_KEYS, URL_CHARS


class TextEncoder(nn.Module):  # DistilBERT, [CLS] 768
    out_dim = 768

    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BERT)

    def forward(self, b, tokens=False):
        h = self.enc(input_ids=b["ids"], attention_mask=b["mask"]).last_hidden_state
        return h if tokens else h[:, 0]


class MarkupEncoder(nn.Module):  # MarkupLM: text + XPath of each HTML node, [CLS] 768
    out_dim = 768

    def __init__(self):
        super().__init__()
        # checkpoint is stored in fp16; pooler is not pretrained, so [CLS] is used like in DistilBERT
        self.enc = MarkupLMModel.from_pretrained(MARKUPLM, dtype=torch.float32, add_pooling_layer=False)

    def forward(self, b, tokens=False):
        return self.enc(**{k: b[k] for k in MARKUP_KEYS}).last_hidden_state[:, 0]


class ImageEncoder(nn.Module):  # ImageNet CNN without classifier: pooled vector or 7x7 patch tokens
    def __init__(self, name="resnet50"):
        super().__init__()
        if name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.body, self.out_dim = nn.Sequential(*list(m.children())[:-2]), 2048
        else:
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.body, self.out_dim = m.features, 1280

    def forward(self, b, tokens=False):
        f = self.body(b["img"])                       # (B, C, 7, 7)
        return f.flatten(2).transpose(1, 2) if tokens else f.mean((2, 3))


class URLCharCNN(nn.Module):  # character-level 1D CNN over the URL, trained from scratch
    def __init__(self, emb=32, out_dim=256):
        super().__init__()
        self.emb = nn.Embedding(len(URL_CHARS) + 2, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, 128, k, padding=k // 2) for k in (3, 5, 7)])
        self.proj, self.out_dim = nn.Linear(3 * 128, out_dim), out_dim

    def forward(self, b, tokens=False):
        e = self.emb(b["url"]).transpose(1, 2)
        return self.proj(torch.cat([F.relu(c(e)).amax(dim=2) for c in self.convs], dim=1))


class LateFusion(nn.Module):
    def __init__(self, dims, hidden=512):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(sum(dims), hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 2))

    def forward(self, feats):
        return self.head(torch.cat(feats, dim=1))


class GatedFusion(nn.Module):
    def __init__(self, dims, d=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(k, d) for k in dims])
        self.gate = nn.Linear(d * len(dims), len(dims))
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.3), nn.Linear(d, 2))

    def forward(self, feats, present):
        z = torch.stack([p(f) for p, f in zip(self.proj, feats)], dim=1)   # (B, M, d)
        logits = self.gate(z.flatten(1)).masked_fill(present == 0, -1e4)
        w = torch.softmax(logits, dim=1)
        return self.head((w.unsqueeze(-1) * z).sum(1)), w


def modality_dropout(feats, p, training):
    keep = torch.ones(feats[0].size(0), len(feats), device=feats[0].device)
    if training:
        keep = (torch.rand_like(keep) > p).float()
        keep[keep.sum(1) == 0, 0] = 1                                       # keep at least one modality
    return [f * keep[:, i:i + 1].to(f.dtype) for i, f in enumerate(feats)], keep


class CrossAttentionFusion(nn.Module):  # text tokens <-> image patch tokens
    def __init__(self, txt_dim, img_dim, d=256, heads=4):
        super().__init__()
        self.t, self.v = nn.Linear(txt_dim, d), nn.Linear(img_dim, d)
        self.t2v = nn.MultiheadAttention(d, heads, batch_first=True)
        self.v2t = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm_t, self.norm_v = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d, 2))

    def forward(self, txt, mask, img):            # (B, Lt, 768), (B, Lt), (B, 49, 2048)
        t, v = self.t(txt), self.v(img)
        t2, _ = self.t2v(t, v, v)
        v2, _ = self.v2t(v, t, t, key_padding_mask=(mask == 0))
        t, v = self.norm_t(t + t2), self.norm_v(v + v2)
        m = mask.unsqueeze(-1).to(t.dtype)
        t_pool = (t * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([t_pool, v.mean(1)], dim=1))


SPECS = {  # model name -> (encoders, fusion)
    "text": (["text"], "late"), "image": (["image"], "late"), "url": (["url"], "late"),
    "late": (["text", "image"], "late"), "late3": (["text", "image", "url"], "late"),
    "gated": (["text", "image", "url"], "gated"), "markup": (["markup", "image"], "late"),
    "xattn": (["text", "image"], "xattn"),
}
LR = {"text": 2e-5, "markup": 2e-5, "image": 1e-4, "url": 1e-3, "head": 1e-3}


class Net(nn.Module):
    def __init__(self, name, backbone="resnet50"):
        super().__init__()
        self.mods, self.fusion = SPECS[name]
        make = {"text": TextEncoder, "markup": MarkupEncoder, "url": URLCharCNN,
                "image": lambda: ImageEncoder(backbone)}
        self.enc = nn.ModuleDict({m: make[m]() for m in self.mods})
        dims = [self.enc[m].out_dim for m in self.mods]
        if self.fusion == "late":
            self.head = LateFusion(dims)
        elif self.fusion == "gated":
            self.head = GatedFusion(dims)
        else:
            self.head = CrossAttentionFusion(*dims)

    def forward(self, b):
        """Returns (logits, gate weights or None)."""
        if self.fusion == "xattn":
            return self.head(self.enc["text"](b, True), b["mask"], self.enc["image"](b, True)), None
        feats = [self.enc[m](b) for m in self.mods]
        if self.fusion == "gated":
            feats, present = modality_dropout(feats, 0.2, self.training)
            return self.head(feats, present)
        return self.head(feats), None

    def param_groups(self):
        return ([{"params": e.parameters(), "lr": LR[m]} for m, e in self.enc.items()]
                + [{"params": self.head.parameters(), "lr": LR["head"]}])
