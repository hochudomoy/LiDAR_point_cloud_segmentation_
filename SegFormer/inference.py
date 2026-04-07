from transformers import SegformerForSemanticSegmentation
import torch
import torch.nn as nn
from range_image import range_image, spherical_projection

def load_model(type):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )

    old_patch = model.segformer.encoder.patch_embeddings[0].proj
    new_patch_embed = nn.Conv2d(5, 32, kernel_size=7, stride=4, padding=3)

    with torch.no_grad():
        new_patch_embed.weight[:, :3, :, :] = old_patch.weight
        new_patch_embed.weight[:, 3:, :, :].zero_()

    model.segformer.encoder.patch_embeddings[0].proj = new_patch_embed
    model.decode_head.classifier = nn.Conv2d(256, 2, kernel_size=1)

    if (type=='ground'):
        model.load_state_dict(torch.load(
        "C:\\Users\\User\\LiDAR_point_cloud_segmentation\\SegFormer\\SegFormer_ground.pth",
        map_location='cpu'
    ))
    else:
        model.load_state_dict(torch.load(
        "C:\\Users\\User\\LiDAR_point_cloud_segmentation\\SegFormer\\SegFormer_road.pth",
        map_location='cpu'
    ))
    model.eval()
    return model

def SegFormer(lidar_df, model):
    ri = range_image(lidar_df)
    input_tensor = torch.tensor(ri).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(input_tensor)

    pred_mask = torch.argmax(pred.logits, dim=1).squeeze(0).cpu().numpy()

    u, v, r = spherical_projection(lidar_df, H=16, W=512)
    point_labels = pred_mask[v, u]

    probs = torch.softmax(pred.logits, dim=1)
    ground_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy()
    point_prob = ground_prob[v, u]

    return point_labels, point_prob

