import torch
from torch import nn
import dinosaur
import slot_attention
from torchvision.transforms import Resize
import timm


class ResDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=3, pool=None):
        super().__init__()
        block = []
        for i in range(n_layers):
            block.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                         kernel_size=3, padding=1))
            block.append(nn.LeakyReLU())
        self.combine = nn.Conv2d(
            in_channels+out_channels, out_channels, 3, padding=1)
        self.block = nn.Sequential(*block)
        if pool is None:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = pool

    def forward(self, x):
        y = self.block(x)
        x = torch.cat([x, y], dim=1)
        x = self.combine(x)
        x = self.pool(x)
        return x


class OCClip(nn.Module):
    def __init__(self, device, clip_dim=768, num_transformer_layers=8, allow_dropout_everything=False, oc_type="DINOSAUR", feature_extractor="resnet18", load_pretrained_oc=True):
        super().__init__()

        self.allow_dropout_everything = allow_dropout_everything

        self.device = device

        if oc_type == "SA":
            self.resolution = (128, 128)
            self.sa = slot_attention.SlotAttentionAutoEncoder(
                self.resolution, 7, 3, 64)
            if load_pretrained_oc:
                self.sa.load_state_dict(torch.load(
                    f'../models/slotattention_clever.ckpt')['model_state_dict'])
        else:
            encoder_name = "vit_base_patch8_224_dino"
            slot_dim = 256
            n_slots = 7
            self.resolution = (224, 224)
            self.sa = dinosaur.DINOSAUR(n_slots, 3, slot_dim, 2048, encoder_name=encoder_name,
                                        use_transformer=True).to(device)
            if load_pretrained_oc:
                # self.sa.load_state_dict(torch.load(
                #     f'../models/model_vit_base_patch8_224_dino_dataset.COCO_transformerTrue.ckpt')['model_state_dict'])
                self.sa.load_state_dict(torch.load(
                    f'../models/dinosaur_bedroom.ckpt')['model_state_dict'])

        for param in self.sa.parameters():
            param.requires_grad = False

        self.resize = Resize(self.resolution)

        self.features = timm.create_model(
            feature_extractor, pretrained=True, num_classes=0).to(device)
        d = self.features(torch.empty(1, 3, 224, 224, device=device)).shape[-1]
        self.features_to_clipdim = nn.Linear(d, clip_dim-1)

        transformer_layer = nn.TransformerEncoderLayer(
            clip_dim, 8, batch_first=True, dim_feedforward=2*clip_dim)
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_transformer_layers)

        self.slot_randomness = torch.Generator(device)

        self.t = torch.nn.parameter.Parameter(torch.tensor([0.007]))

        self.dropout_distribution = torch.distributions.beta.Beta(
            1e-8, 1e5)

    def get_masks(self, x):
        with torch.no_grad():
            img = self.resize(x)
            if self.slot_randomness is not None:
                self.slot_randomness.manual_seed(1)
            *_, image_masks, slots = self.sa(
                img, generator=self.slot_randomness)

            if len(image_masks.shape) == 4:
                b, s, n, _ = image_masks.shape
                wh = int(n**0.5)
                image_masks = image_masks.reshape(b, s, wh, wh, 1)

            b, s, w, h, _ = image_masks.shape

            image_masks = image_masks.reshape(b*s, 1, w, h)
            image_masks = self.resize(image_masks)

            device = image_masks.device

            w, h = self.resize.size
            dropout_prob = self.dropout_distribution.sample(
                (b, 1, 1, 1, 1)).to(device)

            dropout_mask = self.get_dropout_mask(
                (b, s, 1, 1, 1), dropout_prob, device, image_masks.dtype)

            image_masks = image_masks.reshape(b, s, 1, w, h) * dropout_mask
        return slots, image_masks

    def get_dropout_mask(self, shape, probs, device, dtype):
        b, *rest_shape = shape
        masks = []
        for prob in probs:
            while True:
                dropout_mask = (torch.rand(rest_shape, device=device) >
                                prob).to(dtype)
                invalid = ~(dropout_mask.any())
                if not invalid or self.allow_dropout_everything:
                    masks.append(dropout_mask)
                    break
        masks = torch.stack(masks)
        return masks

    def features_to_clip(self, features):
        features = self.features_to_clipdim(features)
        b, s, d = features.shape
        # add start_of_sentence token
        features = torch.cat([
            torch.zeros((b, 1, d+1)).to(self.device),
            torch.cat([
                torch.zeros((b, s, 1)).to(self.device),
                features
            ], dim=-1)
        ], dim=-2)
        features[:, 0, 0] = 1

        cliplike = self.transformer(features)[
            :, 0, :]  # use transformed start-token
        # cliplike has shape [batch_size, clip_dim]

        return cliplike

    def forward(self, x):
        slots, masks = self.get_masks(x)
        # masks has shape [batch_size, num_slots, num_channels=1, width, height]
        x = self.resize(x)
        b, c, w, h = x.shape
        x = x.reshape(b, 1, c, w, h)
        # x has shape [batch_size, 1, num_channels=3, width, height]

        masked_images = x*masks
        # masked_images has shape [batch_size, num_slots, num_channels=3, width, height]
        b, s, c, w, h = masked_images.shape
        masked_images = masked_images.reshape(b*s, c, w, h)

        masked_features = self.features(masked_images)
        # masked_features has shape [batch_size*num_slots, clip_dim-1]
        masked_features = masked_features.reshape(b, s, -1)

        cliplike = self.features_to_clip(masked_features)

        return cliplike, self.t
