import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load(repo_or_dir="intel-isl/MiDaS", model="transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
file_name=r"/home/muahmmad/projects/Image_enhancement/Enhancement_Dataset/7393_NF2_f000010.jpg"
img = cv2.imread(filename=file_name)
input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
output = prediction.cpu().numpy()
print(output)
print(output.shape)
cv2.imshow(winname="orignal RGB image",mat=img)
plt.imshow(output,cmap='hot_r')
plt.show()
cv2.imwrite(filename=r"midas_results.png",img=output)

cv2.waitKey(0)
cv2.destroyAllWindows()