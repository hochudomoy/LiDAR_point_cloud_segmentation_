import torch
import torch.nn as nn
from qai_hub_models.models.salsanext import Model
from range_image import range_image, spherical_projection

def load_model():
    model = Model.from_pretrained()

    model.model.module.logits = nn.Conv2d(32, 2, kernel_size=1)

    model.load_state_dict(torch.load(
        "C:\\Users\\User\\LiDAR_point_cloud_segmentation\\SalsaNext\\salsanext_ground.pth",
        map_location='cpu'
    ))

    model.eval()
    return model
def SalsaNext(lidar_df, model):
    ri = range_image(lidar_df)
    input_tensor = torch.tensor(ri).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(input_tensor)

    pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    u, v, r = spherical_projection(lidar_df, H=64, W=2048)
    point_labels = pred_mask[v, u]

    probs = torch.softmax(pred, dim=1)
    ground_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy()
    point_prob = ground_prob[v, u]

    return point_labels, point_prob
