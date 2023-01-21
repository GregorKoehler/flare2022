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

from copy import deepcopy

import numpy as np
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props_v2
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v24 import ExperimentPlanner3D_v24
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *


class ExperimentPlanner3D_v25(ExperimentPlanner3D_v24):
    """
    - 3d lowres is back to 50 percentile
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.5"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.5_plans_3D.pkl")

    def plan_experiment(self):
        return ExperimentPlanner3D_v21.plan_experiment(self)