from evals.models.dino import DINO
from evals.models.stablediffusion import DIFT
from evals.models.clip import CLIP
from evals.models.sam import SAM
from evals.models.radio import RADIO

from PIL import Image
import numpy as np
import torch
from torchvision import transforms as transforms
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from sklearn.decomposition import PCA
import pdb
import random
import os
import open_clip

def PCA_visualize(feature, H, W, return_res=False, pcaed=False, mask=None):

    feature_img_resized = F.interpolate(feature, 
                            size=(H, W), 
                            mode='bilinear', 
                            align_corners=True)
    feature_img_resized = feature_img_resized[0].permute(1, 2, 0)
    feature = feature_img_resized
    if feature.device != torch.device('cpu'):
        feature = feature.cpu()

    if not pcaed:
        pca = PCA(n_components=3)
        tmp_feature = feature.reshape(-1, feature.shape[-1]).detach().numpy()
        pca.fit(tmp_feature)
        pca_feature = pca.transform(tmp_feature)
        for i in range(3): # min_max scaling
            pca_feature[:, i] = (pca_feature[:, i] - pca_feature[:, i].min()) / (pca_feature[:, i].max() - pca_feature[:, i].min())
        pca_feature = pca_feature.reshape(feature.shape[0], feature.shape[1], 3)
    else:
        pca_feature = feature
    print(pca_feature.shape)
    if return_res:
        return pca_feature
    plt.imshow(pca_feature)  # cmap='gray'表示使用灰度颜色映射
    plt.axis('off')
    plt.show()

    
def load_image(url, transform_size):
    img = Image.open(url)
    img = np.array(img)[:, :, :3]
    H, W = img.shape[0], img.shape[1]
    img = Image.fromarray(img)
    rgb_transform = transforms.Compose(
                [
                    transforms.Resize((transform_size, transform_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
    img = rgb_transform(img).to('cuda')
    img = img.unsqueeze(0).detach()
    return img, H, W

def transform_np_image_to_torch(image, transform_size):
    img = np.array(image)[:, :, :3]
    H, W = img.shape[0], img.shape[1]
    img = Image.fromarray(img)
    rgb_transform = transforms.Compose(
                [
                    transforms.Resize((transform_size, transform_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
    img = rgb_transform(img).to('cuda')
    img = img.unsqueeze(0).detach()
    return img, H, W

def get_point_cloud(obs):
    camera_params = obs["camera_param"]
    images = obs["image"]
    camera_dicts = camera_params
    for camera_name in camera_dicts:
        camera_intrinsic = camera_dicts[camera_name]["intrinsic_cv"]
        cam2world_matrix = camera_dicts[camera_name]["cam2world_gl"]
        Rtilt_rot = cam2world_matrix[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        Rtilt_trl = cam2world_matrix[:3, 3]
        cam2_wolrd = np.eye(4)
        cam2_wolrd[:3, :3] = Rtilt_rot
        cam2_wolrd[:3, 3] = Rtilt_trl
        camera_dicts[camera_name]["cam2world"] = cam2_wolrd
        camera_image = images[camera_name]
        camera_rgb = camera_image["rgb"]
        camera_depth = camera_image["depth"]
        point_cloud_world, per_point_rgb, = tanslation_point_cloud(camera_depth, camera_rgb,
                                                                                    camera_intrinsic, cam2_wolrd)
            
        camera_dicts[camera_name]["point_cloud_world"] = point_cloud_world
        camera_dicts[camera_name]["per_point_rgb"] = per_point_rgb
        camera_dicts[camera_name]["rgb"] = camera_rgb
        camera_dicts[camera_name]["depth"] = camera_depth
        camera_dicts[camera_name]["camera_intrinsic"] = camera_intrinsic
        # view_point_cloud_parts(point_cloud=point_cloud_world, mask=seg_mask)
        # view_point_cloud_parts(point_cloud=point_cloud_world, rgb=per_point_rgb)
    return camera_dicts

def tanslation_point_cloud(depth_map, rgb_image, camera_intrinsic, cam2world_matrix):
    depth_map = depth_map.reshape(depth_map.shape[0], depth_map.shape[1])
    rows, cols = depth_map.shape[0], depth_map.shape[1]
    
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    z = depth_map
    x = (u - camera_intrinsic[0][2]) * z / camera_intrinsic[0][0]
    y = (v - camera_intrinsic[1][2]) * z / camera_intrinsic[1][1]
    points = np.dstack((x, y, z))
    per_point_xyz = points.reshape(-1, 3)
    per_point_rgb = rgb_image.reshape(-1, 3)
    # view_point_cloud_parts(per_point_xyz, actor_seg)
    point_xyz = [per_point_xyz]
    point_rgb = [per_point_rgb]
    # print('!', point_xyz[0].shape, point_rgb[0].shape)

    if len(point_xyz) > 0:
        pcd_camera = np.concatenate(point_xyz)
        point_rgb = np.concatenate(point_rgb)
        pcd_world = pc_camera_to_world(pcd_camera, cam2world_matrix)
        return pcd_world, point_rgb
    else:
        return None, None

def pc_camera_to_world(pc, extrinsic):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    pc = (R @ pc.T).T + T
    return pc


# ===== Model =====
# model = CLIP(arch="ViT-B-16").to('cuda')
model = SAM(arch="vit_h", device='cuda')
# model = RADIO().to('cuda')
# model = DIFT().to('cuda')
# model = DINO(dino_name="dinov2", model_name='vits14').to('cuda')

print('model is loaded')
# ================
url = '../huawei'
# file_type = 'image'
file_type = 'pickle'
cnt = 0

with open('../ctx/192.pickle', 'rb') as file:
    data = pickle.load(file)
camera_list = list(data['image'].keys())

vis_depth = False

# =================


rgb_list = []
feature = []
if file_type == 'pickle':
    with torch.no_grad():
        for camera in camera_list:
            image = data['image'][camera]['rgb']
            
            rgb_list.append(image)
            img, H, W = transform_np_image_to_torch(image, transform_size=1344)
            res = model(img)
            feature.append(res.cpu())
    cnt = len(camera_list)
else:
    with torch.no_grad():
        folder_path = url
        for img_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_path)
            img = Image.open(img_path).convert('RGB')
            image = np.array(img)
            rgb_list.append(image)
            img, H, W = transform_np_image_to_torch(image, transform_size=1344)
            res = model(img)
            feature.append(res.cpu())
    cnt = len(os.listdir(folder_path))

feature = np.array(feature)
new_order = (0, 1, 3, 4, 2)
new_feature = np.transpose(feature, new_order)
orig_shape = new_feature.shape
new_feature = new_feature.reshape(-1, new_feature.shape[-1])
print(f'orig: {feature.shape}, new: {new_feature.shape}')

pca = PCA(n_components=3)
pca.fit(new_feature)
pca_features = pca.transform(new_feature)   

pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                     (pca_features[:, 0].max() - pca_features[:, 0].min())

pca_features_bg = pca_features[:, 0] < -1
pca_features_fg = ~pca_features_bg



pca.fit(new_feature[pca_features_fg])
pca_features_fg = pca_features_fg.reshape(orig_shape[:-1] + (1,))
pca_features_bg = pca_features_bg.reshape(orig_shape[:-1] + (1,))

pca_features = pca.transform(new_feature)

# show mask
for i in range(cnt):
    plt.subplot(3, 2, i+1)
    plt.imshow(pca_features_bg[i][0])
    plt.axis('off')
plt.show()


# show raw image
if file_type == 'pickle':
    i = 0
    for camera in camera_list:
        image = data['image'][camera]['rgb']
        rgb = torch.Tensor(np.array(image))
        # res = PCA_visualize(rgb, H, W, return_res=True, pcaed=True)
        plt.subplot(3, 2, i+1)
        plt.imshow(rgb / 255)
        plt.axis('off')
        i += 1
    plt.show()
else:
    i = 0
    for img_path in os.listdir(folder_path):
        path = os.path.join(folder_path, img_path)
        img = Image.open(path).convert('RGB')
        rgb = torch.Tensor(np.array(img))
        # res = PCA_visualize(rgb, H, W, return_res=True, pcaed=True)
        plt.subplot(3, 2, i+1)
        plt.imshow(rgb / 255)
        plt.axis('off')
        i += 1
    plt.show()

for i in range(3):
    # min_max scaling
    pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())


feature_rgb = pca_features.reshape(orig_shape[:-1] + (3,))
feature_rgb = np.transpose(feature_rgb, (0, 1, 4, 2, 3))
print(feature_rgb.shape)


if file_type == 'pickle':
    for i in range(len(camera_list)):
        camera = camera_list[i]
        image = data['image'][camera]['rgb']
        H, W = image.shape[:2]
        rgb = torch.Tensor(feature_rgb[i])
        tmp = np.transpose(pca_features_bg[i][0], (2, 0, 1))

        rgb[0, :, tmp[0]] = 0
        res = PCA_visualize(rgb, H, W, return_res=True, pcaed=True)
        data['image'][camera]['rgb'] = res
        plt.subplot(3, 2, i+1)
        plt.title(camera)
        plt.imshow(res)
        plt.axis('off')
    plt.show()

else:
    folder_path = url
    i = 0
    for img_path in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_path)
        img = Image.open(img_path).convert('RGB')
        image = np.array(img)
        H, W = image.shape[:2]
        rgb = torch.Tensor(feature_rgb[i])
        tmp = np.transpose(pca_features_bg[i][0], (2, 0, 1))

        # rgb[0, :, tmp[0]] = 0
        
        res = PCA_visualize(rgb, H, W, return_res=True, pcaed=True)
        plt.subplot(3, 2, i+1)
        plt.imshow(res)
        plt.axis('off')
        i += 1
    plt.show()


if file_type == 'pickle':
    pcd_dict = get_point_cloud(data)

    pcd = np.empty((0, 3), dtype=np.float16)
    pcd_rgb = np.empty((0, 3), dtype=np.float16)
    for camera in camera_list:
        tmp_xyz = pcd_dict[camera]['point_cloud_world']
        tmp_rgb = pcd_dict[camera]['per_point_rgb']

        # crop ground
        bool_mask = (tmp_xyz[:, 0] <= 5 ) & (tmp_xyz[:, 0] >= -0.37)
        masked_idx = np.where(bool_mask)[0]
        masked_xyz = tmp_xyz[masked_idx]
        masked_rgb = tmp_rgb[masked_idx]

        # crop ground
        bool_mask = (masked_xyz[:, 1] <= 1) & (masked_xyz[:, 1] >= -0.3)
        masked_idx = np.where(bool_mask)[0]
        masked_xyz = masked_xyz[masked_idx]
        masked_rgb = masked_rgb[masked_idx]
                                
        t = 0.1 # crop ground
        # crop ground
        bool_mask = masked_xyz[:, 2] > t
        masked_idx = np.where(bool_mask)[0]
        masked_xyz = masked_xyz[masked_idx]
        masked_rgb = masked_rgb[masked_idx]

        pcd = np.concatenate((pcd, masked_xyz), axis=0)   
        pcd_rgb = np.concatenate((pcd_rgb, masked_rgb), axis=0) 



    for i in range(3):
        # min_max scaling
        pcd_rgb[:, i] = (pcd_rgb[:, i] - pcd_rgb[:, i].min()) / (pcd_rgb[:, i].max() - pcd_rgb[:, i].min())

    print(pcd.shape)

    pcd = pcd
    idx = random.sample(range(0, pcd.shape[0]), 10000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcd[idx, 0], pcd[idx, 1], pcd[idx, 2], c=pcd_rgb[idx], s=1, marker='o')
    # 设置图表标题
    ax.set_title('3D Point Cloud Rendering')
    ax.set_axis_off()
    # 显示图表
    plt.show()