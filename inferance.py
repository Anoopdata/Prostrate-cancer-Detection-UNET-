import numpy as np
import SimpleITK as sitk
from utils import preprocess_image







import torch


# Define the model architecture
model_name = "unet"  # Replace with the model you trained (e.g., attention_unet, segresnet)
model = get_model(model_name)

# Load the traisausagened weights
model_path = "/home/alphadroid/Music/prostate/Prostate-Segmentation-main/results/training/unet/fold_0/best_net.sausage"  # Replace with your model path
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



# Load the new data
input_file = "data/000/data.nrrd"  # Path to a new image
image = sitk.ReadImage(input_file)
image_array = sitk.GetArrayFromImage(image)

# Preprocess the image (normalize, resample if required)
preprocessed_image = preprocess_image(image_array)

# Convert to PyTorch tensor and move to device
input_tensor = torch.from_numpy(preprocessed_image).unsqueeze(0).unsqueeze(0).float().to(device)

# Perform inference
with torch.no_grad():
    prediction = model(input_tensor)
    prediction = prediction.squeeze().cpu().numpy()

# Post-process the prediction (e.g., thresholding, resizing to original spacing)
prediction_binary = (prediction > 0.5).astype(np.uint8)

# Save the prediction
output_file = "/home/alphadroid/Music/prostate/Prostate-Segmentation-main/inferance/data"
output_image = sitk.GetImageFromArray(prediction_binary)
output_image.CopyInformation(image)  # Retain original image metadata
sitk.WriteImage(output_image, output_file)
