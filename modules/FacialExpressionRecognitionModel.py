from torch import nn
from modules.RawFeatureExtractor import RawFeatureExtractor
from modules.LBPFeatureExtractor import LBPFeatureExtractor
from modules.AttentionalSelectiveFusion import AttentionalSelectiveFusion
from modules.PatchEmbedding import PatchEmbedding
from modules.VisionTransformer import VisionTransformer

class FacialExpressionRecognitionModel(nn.Module):
    def __init__(self, num_classes, emb_dim=128, patch_size=1,
                 transformer_depth=4, n_heads=8, mlp_dim=256):
        super(FacialExpressionRecognitionModel, self).__init__()
        self.raw_extractor = RawFeatureExtractor()
        self.lbp_extractor = LBPFeatureExtractor()
        self.asf = AttentionalSelectiveFusion(in_channels=64)
        self.patch_embed = PatchEmbedding(in_channels=64, emb_dim=emb_dim, patch_size=patch_size)
        self.transformer = VisionTransformer(emb_dim=emb_dim, depth=transformer_depth,
                                             n_heads=n_heads, mlp_dim=mlp_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        raw_feat = self.raw_extractor(x)
        lbp_feat = self.lbp_extractor(x)
        fused_feat = self.asf(raw_feat, lbp_feat)
        patches = self.patch_embed(fused_feat)
        transformer_out = self.transformer(patches)
        logits = self.classifier(transformer_out)
        return logits