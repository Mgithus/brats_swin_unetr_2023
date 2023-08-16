'''set up the config for brats dataset, train and test folders names and
some parameters'''
import os
import torch
import monai
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from functools import partial

class Config:
    class newGlobalConfigs:
        root_dir = "/content/drive/MyDrive/20_samples_2023data"
        train_root_dir = root_dir + "/train_ds"
        test_root_dir = root_dir + "/test_ds"
        path_to_xlsx = root_dir + "/BraTS2023_2017_GLI_Mapping.xlsx"
        pretrained_model = ""
        survival_info_df = ""
        a_test_patient = "BraTS-GLI-00016-000" # change it to other patients, if needed
        full_patient_path = train_root_dir +"/" + a_test_patient
        name_mapping_df = path_to_xlsx
        seed = 50
        json_file = root_dir + "/short_2023data.json"
        
        class OtherPC:
            root_dir = "/content/drive/MyDrive"
            train_root_dir = root_dir + "/short_2021data"
            path_to_csv = "/content/drive/MyDrive/20_samples_2023data/BraTS2023_2017_GLI_Mapping.xlsx"
            a_test_patient = "BraTS2021_00002" # change it to other patients, if needed
            full_patient_path = train_root_dir +"/" + a_test_patient
            name_mapping_df = path_to_csv
            seed = 50
            json_file = root_dir + "/brats21_folds.json"

        class swinUNetCongis:
            roi = (128, 128, 128)
            fold = 1
            max_epochs = 100
            infer_overlap = 0.5
            val_every = 2
            class training_cofigs:
                roi = (128, 128, 128)
                num_workers = 2
                batch_size = 1
                sw_batch_size = 1
                infer_overlap = 0.6
                dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
                post_simgoid = Activations(sigmoid= True)
                post_pred = AsDiscrete(argmax= False, threshold= 0.5)
                dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, 
                                      get_not_nans=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = SwinUNETR(
                                    img_size=roi,
                                    in_channels=4,
                                    out_channels=3,
                                    feature_size=48,
                                    drop_rate=0.0,
                                    attn_drop_rate=0.0,
                                    dropout_path_rate=0.0,
                                    use_checkpoint=True,
                                ).to(device)
                
                model_inferer = partial(
                                        sliding_window_inference,
                                        roi_size=[roi[0], roi[1], roi[2]],
                                        sw_batch_size=sw_batch_size,
                                        predictor=model,
                                        overlap=infer_overlap)
                learning_rate = 1e-4
                weight_decay = 1e-5
                max_epochs = 100
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                              weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)





