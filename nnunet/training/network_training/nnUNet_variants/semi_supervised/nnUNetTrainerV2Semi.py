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


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.network_architecture.generic_modular_UNet import PlainConvUNet
from nnunet.training.data_augmentation.data_augmentation_insaneDA import (
    get_insaneDA_augmentation,
)
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params,
    get_patch_size,
    default_3D_augmentation_params,
)
from nnunet.training.dataloading.dataset_loading import (
    load_dataset,
    DataLoader3D,
    DataLoader2D,
    unpack_dataset,
)
from nnunet.training.network_training.nnUNet_variants.semi_supervised.dataloader import (
    DataLoader3DSemi,
)
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


class nnUNetTrainerV2Semi(nnUNetTrainerV2):
    """
    Trainer to support an additional dataset to be loaded alongside the main dataset
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        additional_dataset_directory=None,
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
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.additional_dataset_directory = additional_dataset_directory

        self.pin_memory = True

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)
        self.additional_dataset = load_dataset(self.additional_dataset_directory)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3DSemi(
                self.dataset_tr,
                self.additional_dataset,
                0.75,  # additional-data batch share (3/4 from additional dataset each batch here)
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
            dl_val = DataLoader3D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        else:
            dl_tr = DataLoader2D(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
            dl_val = DataLoader2D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        return dl_tr, dl_val

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced TrainverV2's moreDA with get_insaneDA_augmentation (Gregor)
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
                    if self.additional_dataset_directory is not None:
                        unpack_dataset(self.additional_dataset_directory)
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
