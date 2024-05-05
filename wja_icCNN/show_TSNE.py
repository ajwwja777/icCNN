import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

path0 = '../Data/iccnn/basic_fmap/vgg_download/13_vgg_bird_iccnn.npz'#bird_iccnn(下载的)
path1 = '../Data/iccnn/vgg16/16_vgg_voc_multi_iccnn_200.npz'#multi_iccnn
path2 = '../Data/iccnn/basic_fmap/vgg/13_vgg_bird_ori.npz'#my_bird_ori
path3 = '../Data/iccnn/basic_fmap/vgg_download/vgg_voc_bird_lame1_c5_ep2499.npz'#bird_iccnn(论文用)
path4 = '../Data/iccnn/densenet161/161_densenet_voc_multi_iccnn.npz'#multi_iccnn
path5 = '../Data/iccnn/basic_fmap/vgg_download/50_resnet_bird_iccnn.npz'#bird_iccnn(下载的)

ori_path1 = '../Data/iccnn/vgg16/16_vgg_voc_multi_ori.npz'#ori_iccnn
ori_path4 = '../Data/iccnn/densenet161/161_densenet_voc_multi_ori.npz'#ori_iccnn

# 加载特征向量
data = np.load(path5)
all_feature = data['f_map']
print(all_feature.shape)

# all_feature = all_feature.reshape(421, -1)

# all_feature = (all_feature.reshape(all_feature.shape[0], all_feature.shape[1] * all_feature.shape[2] * all_feature.shape[3]))

# 平均池化降维，将特征向量从 (6, 512, 14, 14) 降维到 (6, 512)
all_feature = np.mean(all_feature, axis=(2, 3))
print(all_feature.shape)

# 执行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
all_feature = tsne.fit_transform(all_feature)
print(all_feature.shape)

# 可视化
plt.scatter(all_feature[:, 0], all_feature[:, 1], s=1)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Pooled Feature Vectors')
plt.show()