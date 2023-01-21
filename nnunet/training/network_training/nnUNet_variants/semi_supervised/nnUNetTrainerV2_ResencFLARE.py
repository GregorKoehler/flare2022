from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import nnUNetTrainerV2_ResencUNet


class nnUNetTrainerV2_ResencFLARE(nnUNetTrainerV2_ResencUNet):
    """
    Oversample fg a bit more.
    """
    def __init__(
        self,
        plans_file,
        fold,
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
        self.oversample_foreground_percent = 0.9
