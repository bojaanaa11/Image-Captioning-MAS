embed_size = 512
hidden_size = 512

batch_size = 32
num_epochs = 1000

cnn_encoder_learning_rate = 1e-6
transformer_decoder_learning_rate = 1e-5
transformer_encoder_learning_rate = 1e-5

max_length = 30
freq_threshold = 5
patience = 6
no_improve = 0

cnn_encoder_linear_dropout_rate = 0.5
embed_dropout_rate = 0.5
transformer_decoder_dropout = 0.3
transformer_encoder_dropout = 0.3

cnn_encoder_max_grad_clip_norm = 2.0
transformer_decoder_max_grad_clip_norm = 2.0

weight_decay = 1e-4

transfomer_decoder_nhead = 8
transfomer_encoder_nhead = 8

transformer_decoder_num_layers = 8
transformer_encoder_num_layers = 8
cnn_encoder_num_layers = 6

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision import models
transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()