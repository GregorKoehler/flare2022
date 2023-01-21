#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
import torch
from torch import nn

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.data_augmentation.data_augmentation_insaneDA import (
    get_insaneDA_augmentation,
)
from nnunet.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params,
    get_patch_size,
    default_3D_augmentation_params,
)
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.nd_softmax import softmax_helper

from nnunet.network_architecture.generic_modular_UNet import PlainConvUNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork


class nnUNetTrainerV2insaneDA(nnUNetTrainerV2):
    """
    Same as V2, just using insaneDA.

    Left in initialize_network and setup_DA_params methods for potential future changes. (Gregor)
    """

    def __init__(
        self,
        plans_file,
        fold,
        max_num_epochs=1000,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
        **kwargs
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
        )
        self.max_num_epochs = max_num_epochs
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2**i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array(
                [True]
                + [
                    True if i < net_numpool - 1 else False
                    for i in range(1, net_numpool)
                ]
            )
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans["data_identifier"] + "_stage%d" % self.stage,
            )
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!"
                    )

                self.tr_gen, self.val_gen = get_insaneDA_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                )
                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                    also_print_to_console=False,
                )
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                    also_print_to_console=False,
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        # self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
        #                             len(self.net_num_pool_op_kernel_sizes),
        #                             self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
        #                             dropout_op_kwargs,
        #                             net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
        #                             self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if len(self.net_num_pool_op_kernel_sizes) != len(self.net_conv_kernel_sizes):
            self.net_num_pool_op_kernel_sizes.insert(0, [1, 1, 1])
        self.network = PlainConvUNet(
            input_channels=self.num_input_channels,
            base_num_features=self.base_num_features,
            num_blocks_per_stage_encoder=self.conv_per_stage,
            feat_map_mul_on_downscale=2,
            pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
            conv_kernel_sizes=self.net_conv_kernel_sizes,
            props={
                "conv_op": conv_op,
                "conv_op_kwargs": {"stride": 1, "dilation": 1, "bias": True},
                "dropout_op": dropout_op,
                "dropout_op_kwargs": dropout_op_kwargs,
                "norm_op": norm_op,
                "norm_op_kwargs": norm_op_kwargs,
                "nonlin": net_nonlin,
                "nonlin_kwargs": net_nonlin_kwargs,
            },
            num_classes=self.num_classes,
            num_blocks_per_stage_decoder=None,
            deep_supervision=True,
            upscale_logits=False,
            max_features=512,
            initializer=InitWeights_He(1e-2),
        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(
            list(i)
            for i in 1
            / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
        )[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params["rotation_x"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_y"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_z"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params[
                    "elastic_deform_alpha"
                ] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params[
                    "elastic_deform_sigma"
                ] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params[
                    "rotation_x"
                ]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params["rotation_x"] = (
                    -15.0 / 360 * 2.0 * np.pi,
                    15.0 / 360 * 2.0 * np.pi,
                )
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size[1:],
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )
            self.basic_generator_patch_size = np.array(
                [self.patch_size[0]] + list(self.basic_generator_patch_size)
            )
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size,
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params["selected_seg_channels"] = [0]
        self.data_aug_params["patch_size_for_spatialtransform"] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]["lr"] = poly_lr(
            ep, self.max_num_epochs, self.initial_lr, 0.9
        )
        self.print_to_log_file(
            "lr:", np.round(self.optimizer.param_groups[0]["lr"], decimals=6)
        )
