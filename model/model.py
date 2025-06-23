from transformers import ViTModel, ViTImageProcessorFast, ClapModel, ClapProcessor
import torch
from torchvision import transforms
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageEncoder(nn.Module):
    def __init__(self, model_type, checkpoint):
        super().__init__()
        self.model_type = model_type

        if self.model_type == 'vit':
            # Load the model and the class that the inputs go through
            self.model = ViTModel.from_pretrained(checkpoint, device_map='auto')
            self.preprocessor = ViTImageProcessorFast.from_pretrained(checkpoint)
        else:
            raise Exception('Unsupported model for image encoder')

    def encode(self, data):
        if self.model_type == 'vit':
            inputs = self.preprocessor(images = data, return_tensors="pt", do_normalize=True, do_convert_rgb=True, do_rescale=True, do_resize=True)
            inputs = inputs.to(device)
            outputs = self.model(**inputs)
            return outputs.pooler_output

class AudioEncoder(nn.Module):
    def __init__(self, model_type, checkpoint):
        super().__init__()
        self.model_type = model_type

        if self.model_type == 'clap':
            # Load the model and the class that the inputs go through
            self.model = ClapModel.from_pretrained(checkpoint, device_map='auto')
            self.preprocessor = ClapProcessor.from_pretrained(checkpoint, use_fast=True)
        else:
            raise Exception('Unsupported model for audio encoder')

    def encode(self, sampling_rate, waveforms):
        if self.model_type == 'clap':
            inputs = self.preprocessor(audios=waveforms.numpy(), sampling_rate=sampling_rate, return_tensors='pt')
            inputs = inputs.to(device)
            outputs = self.model.get_audio_features(**inputs)
            return outputs

class MLPMapper(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class PictureToMusicModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_image_encoder()
        self.init_audio_encoder()

        self.image_mapper = MLPMapper(
            input_dim=self.config.image_embedding_size, 
            hidden_dim=self.config.mlp_hidden_size, 
            output_dim=self.config.shared_embedding_size
        )
        if config.has_audio_mapper:
            self.audio_mapper = MLPMapper(
                input_dim=self.config.audio_embedding_size,
                hidden_dim=self.config.mlp_hidden_size,
                output_dim=self.config.shared_embedding_size
            )

        # Learnable temperature
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image_input, audio_input):
        image_emb = self.image_encoder.encode(data=image_input)
        audio_emb = self.audio_encoder.encode(sampling_rate=48000, waveforms=audio_input)

        # print("Image_emb std:", image_emb.std().item())
        # print("Audio_emb std:", audio_emb.std().item())
        image_proj = self.image_mapper(image_emb)
        if self.config.has_audio_mapper:
            audio_proj = self.audio_mapper(audio_emb)
        else:
            audio_proj = audio_emb

        # clamping (restricting) the temperature
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, 0, 100)

        return image_proj, audio_proj, logit_scale

    def init_image_encoder(self):        
        self.image_encoder = ImageEncoder(
            model_type=self.config.image_encoder_type,
            checkpoint=self.config.image_encoder_checkpoint
        )
        
        for param in self.image_encoder.model.parameters():
            param.requires_grad = False
        if not self.config.freeze_image_encoder:
            # Unfreeze only the last couple of layers so we can finetune
            for param in self.image_encoder.model.encoder.layer[-self.config.num_layers_to_unfreeze].parameters():
                param.requires_grad = True

    def init_audio_encoder(self):
        self.audio_encoder = AudioEncoder(
            model_type=self.config.audio_encoder_type, 
            checkpoint=self.config.audio_encoder_checkpoint
        )
        if self.config.freeze_audio_encoder:
            for param in self.audio_encoder.model.parameters():
                param.requires_grad = False
