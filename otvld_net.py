import torch
import torch.nn as nn
import torch.nn.functional as F

class ODConv2d(nn.Module):
    """
    Omni-dimensional Dynamic Convolution
    As described in the diagram: 
    Original Feature Map -> GAP -> FC -> ReLU -> 
    Spatial/Channel/Kernel Attentions -> Dynamic Convolution Kernel -> Output
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, num_experts=4):
        super(ODConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_experts = num_experts

        # Dynamic kernels
        self.weight = nn.Parameter(torch.randn(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        
        # Attention generation
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(1, in_channels // 4)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        # 4 types of attention (Simplified for standard ODConv: Spatial, Channel, Filter/Kernel)
        self.fc_spatial = nn.Conv2d(mid_channels, kernel_size * kernel_size, 1, bias=False)
        self.fc_channel = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.fc_filter = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.fc_expert = nn.Conv2d(mid_channels, num_experts, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Attention Generation
        att = self.gap(x)
        att = self.relu(self.fc1(att))
        
        att_s = self.fc_spatial(att).view(b, 1, 1, 1, self.kernel_size, self.kernel_size).sigmoid()
        att_c = self.fc_channel(att).view(b, 1, 1, self.in_channels // self.groups, 1, 1).sigmoid()
        att_f = self.fc_filter(att).view(b, 1, self.out_channels, 1, 1, 1).sigmoid()
        att_e = self.fc_expert(att).view(b, self.num_experts, 1, 1, 1, 1).softmax(dim=1)
        
        # Weight Assembly
        # weight: (num_experts, out_channels, in_channels//groups, k, k) -> (1, num_experts, out_channels, in_channels//groups, k, k)
        weight = self.weight.unsqueeze(0) 
        
        # Apply attentions
        weight = weight * att_s # Spatial
        weight = weight * att_c # Channel (input)
        weight = weight * att_f # Filter (output)
        
        # Combine experts
        weight = torch.sum(weight * att_e, dim=1) # (b, out_channels, in_channels//groups, k, k)
        
        # Forward pass (batch-wise grouping)
        x = x.view(1, b * c, h, w)
        weight = weight.view(b * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        
        # In ODConv, the weight shape is (batch_size * out_channels, in_channels//groups, k, k)
        # We need groups = batch_size * self.groups in F.conv2d
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=b * self.groups)
        out = out.view(b, self.out_channels, out.size(2), out.size(3))
        
        return out


class TransformerGlobalFeatureFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerGlobalFeatureFusion, self).__init__()
        # Simplified Transformer Encoder for Spatial Features
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim)) # Dummy max length

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # (b, h*w, c)
        
        # Add Positional Embedding
        seq_len = x_flat.size(1)
        x_flat = x_flat + self.pos_embed[:, :seq_len, :]
        
        out = self.transformer(x_flat)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out


class OTVLDNet(nn.Module):
    """
    Paper 2: OTVLD-Net for Lane Detection implementation
    """
    def __init__(self, num_lanes=4):
        super(OTVLDNet, self).__init__()
        
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # OTVLD uses ResNet-18 as the backbone instead of standard convolution blocks
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Feature processing after ResNet-18 (which outputs 512 channels)
        # Using ODConv here to reduce channels and apply omni-dimensional attention
        self.od_conv_fusion = ODConv2d(512, 256, 3, padding=1)
        
        # Vanishing Point Prediction (Auxiliary Head)
        self.vpp_head = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1) # Vanishing point heatmap
        )
        
        # Transformer Global Feature Fusion
        self.transformer_fusion = TransformerGlobalFeatureFusion(embed_dim=256, num_heads=8, num_layers=2)
        
        # Feature Flip Fusion (Simulated via concatenation or direct usage in a real model, we use simple 1x1 convs here to represent MLP/Kernels)
        
        # Lane Prediction Heads
        self.heatmap_kernel = nn.Conv2d(256, num_lanes, 1) # L x C
        self.offset_kernel = nn.Conv2d(256, num_lanes * 2, 1) # L x 2 for dx, dy offsets
        
        self.mlp_vertical_range = nn.Linear(256, num_lanes * 2) # L x 2 for start, end
        self.mlp_object_score = nn.Linear(256, num_lanes) # Probability of lane existing
        
        # Initialize object score to safely output 0 when untrained (prevents random lines)
        nn.init.constant_(self.mlp_object_score.bias, -5.0)
        
    def forward(self, x):
        # x is Input Picture (b, 3, h, w)
        b = x.size(0)
        
        # Dynamic Feature Extraction via ResNet-18
        backbone_features = self.backbone(x)
        features = self.od_conv_fusion(backbone_features)
        
        # VPP Auxiliary
        vp_heatmap = self.vpp_head(features)
        
        # Transformer Global Feature Fusion
        global_features = self.transformer_fusion(features)
        
        # Lane Prediction
        lane_heatmap = self.heatmap_kernel(global_features)
        lane_offset = self.offset_kernel(global_features)
        
        # Pooling for dense prediction representations
        pooled_features = F.adaptive_avg_pool2d(global_features, (1, 1)).view(b, -1)
        
        vertical_range = self.mlp_vertical_range(pooled_features).view(b, -1, 2)
        object_score = torch.sigmoid(self.mlp_object_score(pooled_features))
        
        # In a real deployed application, you post-process these outputs into exact line coordinates.
        return {
            'heatmap': lane_heatmap,
            'offset': lane_offset,
            'vertical_range': vertical_range,
            'object_score': object_score,
            'vp_heatmap': vp_heatmap
        }

if __name__ == '__main__':
    # Test compilation
    print("Testing OTVLD-Net architecture...")
    model = OTVLDNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    out = model(dummy_input)
    print("Inference successful!")
    print(f"Heatmap shape: {out['heatmap'].shape}")
    print(f"VP Heatmap shape: {out['vp_heatmap'].shape}")
