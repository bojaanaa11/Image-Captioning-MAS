
import torch.nn as nn
from torchvision import models

import src.parameters as params
from src.transformer_encoder import TransformerEncoder

once = False

class EncoderCNN(nn.Module):
    def __init__(self, encoder_linear_dropout_rate, num_of_layers_to_unfreeze=-1, unfreeze_mode='top'):
        super().__init__()
        self.num_of_layers_to_unfreeze = num_of_layers_to_unfreeze
        self.unfreeze_mode = unfreeze_mode

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Linear layer to project features from 2048 to embed_size
        self.feature_projection = nn.Linear(2048, params.embed_size)

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            params.embed_size,
            params.transfomer_encoder_nhead,
            params.transformer_encoder_num_layers,
            params.hidden_size,
            params.transformer_encoder_dropout
        )

        self.dropout = nn.Dropout(p=encoder_linear_dropout_rate)

        # Identify parameterized layers (conv/bn layers)
        self.param_layers = [
            idx for idx, layer in enumerate(self.resnet)
            if len(list(layer.parameters())) > 0
        ]

        self.freeze_batchnorm()

        # Fine tune if num_of_layers_to_unfreeze is not -1
        if self.num_of_layers_to_unfreeze != -1:
            self.fine_tune()
        else:
            for p in self.resnet.parameters():
                p.requires_grad = True

    def forward(self, images):
        global once

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Encoder: Images: {images.shape} \n')

        features = self.resnet(images)  # Shape: (batch_size, 2048, H, W)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Encoder: Features from resnet: {features.shape}\n')

        features = features.view(features.size(0), features.size(1), -1)  # Flatten spatial dimensions

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Encoder: Features after view: {features.shape}\n')

        features = features.permute(2, 0, 1)  # Shape: (H*W, batch_size, 2048)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Encoder: Features after permute: {features.shape}\n')

        # Project features to embed_size
        features = self.feature_projection(features)  # Shape: (H*W, batch_size, embed_size)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Encoder: Features after projections: {features.shape}\n')

        # Pass through Transformer Encoder
        features = self.transformer_encoder(features)  # Shape: (H*W, batch_size, embed_size)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Encoder: Features after transformer encoder: {features.shape}\n\n')

            once = True

        features = self.dropout(features)
        return features

    def fine_tune(self):
        if self.num_of_layers_to_unfreeze <= 0 or self.num_of_layers_to_unfreeze > len(self.param_layers):
            raise ValueError(f"num_of_layers_to_unfreeze must be between 1 and {len(self.param_layers)}")

        # Freeze all first
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Determine layers to unfreeze
        if self.num_of_layers_to_unfreeze > 0:
            if self.unfreeze_mode == 'top':
                layers = self.param_layers[-self.num_of_layers_to_unfreeze:]
            elif self.unfreeze_mode == 'bottom':
                layers = self.param_layers[:self.num_of_layers_to_unfreeze]
            else:
                raise ValueError("Use 'top' or 'bottom' mode")

            # Unfreeze selected layers
            for idx in layers:
                for param in self.resnet[idx].parameters():
                    param.requires_grad = True

    def freeze_batchnorm(self):
        for m in self.resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
