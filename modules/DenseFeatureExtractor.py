# DenseFeatureExtractor.py
# Class implementing a feature encoder and decoder based on the args passed to it.
# Code adapted from the official implementation of AdaBins at https://github.com/shariqfarooq123/AdaBins

import os, sys, logging
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Encoder(nn.Module):
    """Wrapper to allow extraction of output from every layer of the base model.
    """
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks' or k == 'features'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))

        return features


class UpSampleWithSkip(nn.Module):
    """Upsamples feature input to dimensions of skip connection, concatenates the two in the channel dimension,
    then runs through two conv/bn/leakyRelu blocks. 
    """
    def __init__(self, input_features, output_features):
        super().__init__()

        self._net = nn.Sequential(nn.Conv2d(input_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, skip_features):
        up_x = F.interpolate(x, size=[skip_features.size(2), skip_features.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, skip_features], dim=1)
        return self._net(f)


class Decoder(nn.Module):
    """Upsampling decoder with skip connections from the encoder."""
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048, mode="features", encoder_name=None, do_final_upscale=False):
        super().__init__()
        features = int(num_features)
        self.encoder_name = encoder_name

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        # Each encoder used has different dimensions for the intermediate features and returns a different total number of features.
        # feature_select chooses which elements in the feature list are used as skip connections
        # Each upsampling layer's input_features need to be changed to match the dimensionality of the incoming skip connections
        if "efficientnet-b5" in self.encoder_name:
            self.feature_select = [4, 5, 6, 8, 11]
            self.up1 = UpSampleWithSkip(input_features=features // 1 + 176, output_features=features // 2)
            self.up2 = UpSampleWithSkip(input_features=features // 2 + 64, output_features=features // 4)
            self.up3 = UpSampleWithSkip(input_features=features // 4 + 40, output_features=features // 8)
            self.up4 = UpSampleWithSkip(input_features=features // 8 + 24, output_features=features // 16)
        elif "efficientnet-b1" in self.encoder_name:
            self.feature_select = [4, 5, 6, 8, 11]
            self.up1 = UpSampleWithSkip(input_features=features // 1 + 112, output_features=features // 2)
            self.up2 = UpSampleWithSkip(input_features=features // 2 + 40, output_features=features // 4)
            self.up3 = UpSampleWithSkip(input_features=features // 4 + 24, output_features=features // 8)
            self.up4 = UpSampleWithSkip(input_features=features // 8 + 16, output_features=features // 16)
        elif "efficientnet-v2-s" in self.encoder_name:
            self.feature_select = [2, 3, 4, 6, 9]
            self.up1 = UpSampleWithSkip(input_features=features // 1 + 160, output_features=features // 2)
            self.up2 = UpSampleWithSkip(input_features=features // 2 + 64, output_features=features // 4)
            self.up3 = UpSampleWithSkip(input_features=features // 4 + 48, output_features=features // 8)
            self.up4 = UpSampleWithSkip(input_features=features // 8 + 24, output_features=features // 16)
        elif "efficientnet-v2-m" in self.encoder_name:
            self.feature_select = [2, 3, 4, 6, 9]
            self.up1 = UpSampleWithSkip(input_features=features // 1 + 176, output_features=features // 2)
            self.up2 = UpSampleWithSkip(input_features=features // 2 + 80, output_features=features // 4)
            self.up3 = UpSampleWithSkip(input_features=features // 4 + 48, output_features=features // 8)
            self.up4 = UpSampleWithSkip(input_features=features // 8 + 24, output_features=features // 16)
        else:
            sys.exit("Error: encoder name not recognised when building decoder.")

        self.final_upscale = None
        if do_final_upscale:
            self.final_upscale = UpSampleWithSkip(input_features=features // 16 + 3, output_features=features // 16)

        self.mode = mode if mode is not None else "features"

        if self.mode == "features":
            # Output to be used with the AdaBins module.
            self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        elif self.mode == "output":
            # A direct depth output.
            self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()


    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = [features[idx] for idx in self.feature_select]

        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)

        if self.final_upscale is not None:
            x_d4 = self.final_upscale(x_d4, features[0])    # features[0] should always be the input image to the encoder.

        out = self.conv3(x_d4)

        return out


class DenseFeatureExtractor(nn.Module):
    """Class implementing a feature encoder and decoder based on the args passed to it.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger = logging.getLogger(__name__)

        self.n_bins = self.args[self.args.model.name].n_bins
        self.num_decoded_channels = 128		# By default this is 128 but will change with different experiments.

        # These two lists get used if args.optimizer.slow_encoder is set (to put a lower learning rate on the encoder.)
        self._encoder_params_module_list = []
        self._non_encoder_params_module_list = []

        # self.ReturnType = namedtuple('ReturnType', ['depth_pred', 'bin_edges'])

        self.logger.info("Building...")

        # Loading efficientnet-b1 or b5 (original version of EfficientNet)
        if "efficientnet-b" in self.args[self.args.model.name].encoder_name:
            if "efficientnet-b5" in self.args[self.args.model.name].encoder_name:
                basemodel_name = 'tf_efficientnet_b5_ap'
                num_out_channels_conv_stem = 48
            elif "efficientnet-b1" in self.args[self.args.model.name].encoder_name:
                basemodel_name = 'tf_efficientnet_b1_ap'
                num_out_channels_conv_stem = 32
            self.logger.info(f'Loading base model ({basemodel_name})...')
            basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
            # Remove unused and final layers
            self.logger.info('Removing unused final layers from encoder (batchnorm2 and act2).')
            basemodel.bn2 = nn.Identity()
            basemodel.act2 = nn.Identity()
            self.logger.info('Removing last two layers (global_pool & classifier).')
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
        
        # Loading efficientnet-v2-s, m, or l.
        elif "efficientnet-v2-" in self.args[self.args.model.name].encoder_name:
            self.logger.info(f'Loading base model ({self.args[self.args.model.name].encoder_name})...')
            if "efficientnet-v2-s" in self.args[self.args.model.name].encoder_name:
                basemodel = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1")
            elif "efficientnet-v2-m" in self.args[self.args.model.name].encoder_name:
                basemodel = torchvision.models.efficientnet_v2_m(weights="IMAGENET1K_V1")

            self.logger.info("Removing final avgpool and classifier layers")
            basemodel.avgpool = nn.Identity()
            basemodel.classifier = nn.Identity()

        assert basemodel is not None, "Error: encoder name not recognised."

        # Building all the blocks
        self.encoder = Encoder(basemodel)
        self._encoder_params_module_list.append(self.encoder)

        if "efficientnet-b5" in self.args[self.args.model.name].encoder_name:
            num_features = 2048
        elif "efficientnet-b1" in self.args[self.args.model.name].encoder_name:
            num_features = 1280
        elif "efficientnet-v2-" in self.args[self.args.model.name].encoder_name:
            num_features = 1280

        self.decoder = Decoder(
                            num_classes=128,
                            num_features=num_features,
                            bottleneck_features=num_features,
                            mode=self.args[self.args.model.name].get("mode"),
                            encoder_name=self.args[self.args.model.name].encoder_name,
                            do_final_upscale=self.args[self.args.model.name].get('do_final_upscale')
                        )

        self._non_encoder_params_module_list.append(self.decoder)

    
    def forward(self, image):
        unet_out = self.encoder(image)
        unet_out = self.decoder(unet_out)
        return unet_out