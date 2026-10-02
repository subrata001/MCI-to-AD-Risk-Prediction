from torch import nn

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

        # Residual gating preserves the input features
        gated_features = x + x * gate_weights

        return gated_features, gate_weights


class RFG_DeepSurv(nn.Module):
    def __init__(self, in_features, hidden_nodes=[64, 32], dropout=0.3):
        super().__init__()

        self.attention_gate = Residual_Feature_Gating(
            feature_dim=in_features,
            reduction=4
        )

        layers = []
        prev_dim = in_features

        for h in hidden_nodes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.risk_net = nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.attention_gate(x)
        risk = self.risk_net(x)
        return risk

