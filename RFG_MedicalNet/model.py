import torch
from torch import nn
import importlib.util
import sys

class Residual_Feature_Gating(nn.Module):
    def __init__(self, feature_dim=512, reduction=4, dropout=0.1):
        super().__init__()

        hidden_dim = feature_dim // reduction

        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate_weights = self.gate(x)

        # Residual gating: preserve original MedicalNet features
        gated_features = x + x * gate_weights

        return gated_features, gate_weights


class RFG_MedicalNet(nn.Module):
    def __init__(self, base_model, feature_dim=512, dropout=0.3):
        super().__init__()

        self.encoder = base_model

        if hasattr(self.encoder, "conv_seg"):
            self.encoder.conv_seg = nn.Identity()

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.aging_gate = Residual_Feature_Gating(
            feature_dim=feature_dim,
            reduction=4,
            dropout=0.1
        )

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_gate=False):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.pool(x)
        features = torch.flatten(x, 1)

        gated_features, gate_weights = self.aging_gate(features)

        age = self.regressor(gated_features).squeeze(1)

        if return_gate:
            return age, gate_weights

        return age


def build_model(resnet_path, pretrained_path, device):
    
    spec = importlib.util.spec_from_file_location("medicalnet_resnet", resnet_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MedicalNet from {resnet_path}")
        
    resnet = importlib.util.module_from_spec(spec)
    sys.modules["medicalnet_resnet"] = resnet
    spec.loader.exec_module(resnet)

    #print([name for name in dir(resnet) if "resnet" in name.lower()])
    #['ResNet', 'resnet10', 'resnet101', 'resnet152', 'resnet18', 'resnet200', 'resnet34', 'resnet50']
    
    base_model = resnet.resnet10(sample_input_W=224,
    sample_input_H=224,
    sample_input_D=224,
    shortcut_type='B',
    no_cuda=False,
    num_seg_classes=1)
    
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    backbone_keys = set(base_model.state_dict()) - {
        k for k in base_model.state_dict() if k.startswith("conv_seg.")}
    if not backbone_keys.intersection(state):
        raise ValueError("Pretrained checkpoint contains no matching backbone keys")
    missing, unexpected = base_model.load_state_dict(state, strict=False)
    print("Pretrained missing keys:", missing, "unexpected keys:", unexpected)
    if any(k in backbone_keys for k in missing):
        raise ValueError("Pretrained checkpoint is missing backbone parameters")
        
    return RFG_MedicalNet(base_model, feature_dim=512).to(device)
