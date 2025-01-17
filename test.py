# DOCUMENT INFORMATION
'''
    Project Name: Prostate Segmentation
    File Name   : test.py
    Code Author : Dejan Kostyszyn and Shrajan Bhandary
    Created on  : 14 March 2021
    Program Description:
        This program predicts the segmentation for the test dataset. It can also generate 
        noisy data to check robustness.
'''

# LIBRARY IMPORTS
import numpy as np
import torch, os, time, models, argparse, nrrd, utils, json
import SimpleITK as sitk
import torchio as tio
import csv
from batchgenerators.utilities.file_and_folder_operations import *
from monai.inferers import sliding_window_inference
from pre_processing.resampler_utils import resample_3D_image

torch.manual_seed(2021)
torch.cuda.empty_cache()

# IMPLEMENTATION

class Predictor():
    def __init__(self, predictor_options=None):
        print("Initializing...")
        self.start_time = time.time()
        self.predictor_options = predictor_options
        self.exp_path = self.predictor_options.exp_path
        self.test_results_path = self.predictor_options.test_results_path
        device_id = 'cuda:' + str(self.predictor_options.device_id)
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        self.data_root = self.predictor_options.data_root
        self.k_fold = self.predictor_options.k_fold
        self.test_files = self.get_test_file_names(data_root=self.data_root)
        self.fold_0_checkpoint = torch.load(os.path.join(self.exp_path, "fold_0", 'best_net.sausage'), 
                                            map_location=self.device)
        self.predictor_options = self.get_train_options(opt=self.predictor_options, 
                                                        model_checkpoint=self.fold_0_checkpoint)
        self.model = models.get_model(opt=self.predictor_options)
        self.model.to(self.device)
        self.current_checkpoint = self.fold_0_checkpoint

    def get_test_file_names(self, data_root):
        test_files = subfiles(data_root, suffix=".nrrd") + subfiles(data_root, suffix=".nii.gz") + subfiles(data_root, suffix=".mhd")
        test_files.sort()
        return test_files

    def get_volume_array_and_info(self, file_location=None):
        old_data_image = sitk.ReadImage(file_location)
        if old_data_image.GetSpacing() != self.predictor_options.voxel_spacing:
            new_spacing = self.predictor_options.voxel_spacing
            new_data_image = resample_3D_image(sitkImage=old_data_image, newSpacing=new_spacing,
                                               interpolation="BSpline", change_spacing=True, change_direction=False)
        else:
            new_data_image = old_data_image

        new_data_array = sitk.GetArrayFromImage(new_data_image).transpose(2, 1, 0)
        new_data_info = {"spacing": new_data_image.GetSpacing(),
                         "direction": new_data_image.GetDirection(),
                         "origin": new_data_image.GetOrigin(),
                        }
        return new_data_array, new_data_info

    def get_train_options(self, opt=None, model_checkpoint=None):
        opt.normalize = model_checkpoint["normalize"]
        opt.patch_shape = model_checkpoint["patch_shape"]
        opt.n_kernels = model_checkpoint["n_kernels"]
        opt.clip = model_checkpoint["clip"]
        opt.seed = model_checkpoint["seed"]
        opt.model_name = model_checkpoint["model_name"]
        opt.voxel_spacing = model_checkpoint["voxel_spacing"]
        opt.input_channels = model_checkpoint["input_channels"]
        opt.output_channels = model_checkpoint["output_channels"]
        opt.no_shuffle = model_checkpoint["no_shuffle"]
        opt.dropout_rate = 0.0  # This is just a formality for model declaration.
        print("Found model: {} that was on volumes with voxel spacing:{} and patch size: {}".
              format(opt.model_name, opt.voxel_spacing, opt.patch_shape))
        return opt

    def save_test_post_processed(self, predicted_array=None, orig_data_path=None, save_location=None,
                                 temp_save_location=None, data_info=None):
        orig_data_name = os.path.split(orig_data_path)[-1]
        predicted_image = sitk.GetImageFromArray(predicted_array.transpose(2, 1, 0))
        predicted_image.SetOrigin(data_info["origin"])
        predicted_image.SetSpacing(data_info["spacing"])
        predicted_image.SetDirection(data_info["direction"])
        if not os.path.exists(temp_save_location):
            os.mkdir(temp_save_location)
        sitk.WriteImage(predicted_image, os.path.join(temp_save_location, orig_data_name))
        predicted_array = np.expand_dims(predicted_array, axis=0)
        orig_data_image = sitk.ReadImage(orig_data_path)
        resample_transform = tio.transforms.Resample(target=orig_data_path, label_interpolation='nearest')
        resampled_pred_image = resample_transform(tio.Image(os.path.join(temp_save_location, orig_data_name), 
                                                             type=tio.LABEL))
        orig_predt_array = resampled_pred_image["data"][0].numpy().astype(np.uint8)
        orig_predt_array = utils.keep_only_largest_connected_component(orig_predt_array)
        orig_predt_array = orig_predt_array.transpose(2, 1, 0)
        predicted_image = sitk.GetImageFromArray(orig_predt_array)
        predicted_image.SetOrigin(orig_data_image.GetOrigin())
        predicted_image.SetSpacing(orig_data_image.GetSpacing())
        predicted_image.SetDirection(orig_data_image.GetDirection())
        if orig_data_image.GetSize() != predicted_image.GetSize():
            print("Shapes don't match")
            print(orig_data_image.GetSize(), predicted_image.GetSize())
        if not os.path.exists(save_location):
            os.mkdir(save_location)
        sitk.WriteImage(predicted_image, os.path.join(save_location, orig_data_name))
        return orig_predt_array

    def test(self, test_completed=False):
        csv_file = os.path.join(self.exp_path, "inference_times.csv")
        with open(csv_file, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["File Name", "Inference Time (seconds)"])
        self.model.eval()
        with torch.no_grad():
            for current_file_path in self.test_files:
                file_name = os.path.split(current_file_path)[-1]
                print(f"\nPredicting label for image: {file_name}")
                start_time = time.time()
                data_array, data_info = self.get_volume_array_and_info(file_location=current_file_path)
                data_array = torch.FloatTensor(data_array)
                data_array = utils.normalize(data_array, opt=self.predictor_options)
                sum_array = torch.zeros_like(data_array).numpy()
                data_array = data_array.to(self.device)
                for current_fold in range(self.k_fold):
                    if current_fold != 0:
                        fold_results_path = os.path.join(self.exp_path, f"fold_{current_fold}")
                        self.current_checkpoint = torch.load(os.path.join(fold_results_path, 'best_net.sausage'), 
                                                             map_location=self.device)
                    self.model.load_state_dict(self.current_checkpoint["model_state_dict"])
                    with torch.cuda.amp.autocast():
                        out_M = sliding_window_inference(
                            inputs=data_array.unsqueeze(0).unsqueeze(0),
                            roi_size=tuple(self.predictor_options.patch_shape),
                            sw_batch_size=2,
                            predictor=self.model,
                            overlap=0.5,
                        )
                    sum_array += utils.get_array(torch.sigmoid(out_M))
                predicted_array = sum_array / float(self.k_fold)
                predicted_array = np.where(predicted_array < 0.5, 0, 1)
                save_location = os.path.join(self.exp_path, "test_predictions_postprocessed")
                temp_save_location = os.path.join(self.exp_path, "test_predictions")
                predicted_array_orig = self.save_test_post_processed(predicted_array=predicted_array, 
                                            orig_data_path=current_file_path,
                                            save_location=save_location,
                                            temp_save_location=temp_save_location,
                                            data_info=data_info)
                unique_vals, unique_counts = np.unique(predicted_array_orig, return_counts=True)
                print(f"Image {file_name} has {unique_vals} different labels with {unique_counts} counts each.")
                end_time = time.time()
                inference_time = end_time - start_time
                print(f"Inference time for {file_name}: {inference_time:.2f} seconds")
                with open(csv_file, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow([file_name, inference_time])
            test_completed = True
        return test_completed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="results/training/unet/", required=False, help="Path to read the results of the cross-validation.")
    parser.add_argument("--test_results_path", type=str, default="results/test/unet/", required=False, help="Path to store the test predictions.")
    parser.add_argument("--data_root", type=str, default="data/test", required=True, help="Path to data.")
    parser.add_argument('--device_id', type=int, default=0, required=False, help='Use the different GPU(device) numbers available. Example: If two Cuda devices are available, options: 0 or 1')
    parser.add_argument('--k_fold', type=int, default=5, required=False, choices = [1,5,8], help='Choose between no crosss validation (ensure separate folders for train and test), or 5-fold crosss validation or 8-fold crosss validation. ')
    opt = parser.parse_args()

    tester = Predictor(predictor_options=opt)
    test_completed = False
    while test_completed is False:
        test_completed = tester.test(test_completed=test_completed)
    print("Prediction is completed")

