# 1 import part
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# 2 Class Gen
def get_generator_block(input_dim, output_dim):

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )

class Generator(nn.Module):
    def __init__(self, z_dim=512, channels=512, height=14, width=14, hidden_dim=128):
        super(Generator, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            nn.Linear(hidden_dim * 4, channels * height * width),  # 将最后一层改为线性层
            nn.Tanh()
        )
    def forward(self, noise):
        x = self.gen(noise)
        # 将大小为[128, 512]的张量转换为大小为[128, 512, height, width]的张量
        x = x.view(-1, 512, 14, 14)
        return x

# 3 Class Disc
class Discriminator(nn.Module):
    def __init__(self, im_dim=(512, 14, 14), hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(im_dim[0], hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, 1, kernel_size=4, stride=2, padding=0)
        )

    def forward(self, image):
        return self.disc(image)

# 4 func get nosise 
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples,z_dim,device=device)

# 5 hyperparameter and so on
z_dim = 512
batch_size = 128
lr = 0.00001
device = 'cpu'

criterion = nn.BCEWithLogitsLoss()
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# 6 func getLoss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device, im_dim):
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    fake = fake.view(num_images, im_dim[0], im_dim[1], im_dim[2])

    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
   
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

# 7 func getLoss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device, im_dim):
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    fake = fake.view(num_images, im_dim[0], im_dim[1], im_dim[2])
    
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    return gen_loss

# 8 func visualize
def visualize_tsne(feature_vectors, title='t-SNE Visualization', xlabel='t-SNE Dimension 1', ylabel='t-SNE Dimension 2'):
    '''
    Function to perform t-SNE dimensionality reduction and visualize the feature vectors.
    Args:
        feature_vectors: 2D numpy array containing feature vectors (samples x features).
        title: Title of the plot (optional).
        xlabel: Label for x-axis (optional).
        ylabel: Label for y-axis (optional).
    '''
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    feature_tsne = tsne.fit_transform(feature_vectors)

    # Visualize t-SNE results
    plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], s=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# 9 dataset
class FeatureDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path)['f_map']  # 加载 npz 文件的特征图数据

    def __len__(self):
        return len(self.data)  # 返回数据集的大小

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float)  # 转换为 PyTorch 张量

# 修改 DataLoader 的数据集为特征图的自定义 Dataset
path = "../../Data/icCNN/16_vgg_bird_iccnn.npzAndOther3/vgg_voc_bird_lame1_c5_ep2499.npz"
dataset = FeatureDataset(path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 10 train
n_epochs = 2
display_step = 5

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
im_dim=(512, 14, 14)
for epoch in range(n_epochs):
  
    # Dataloader returns the batches
    for real in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.to(device)
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device, im_dim)

        ### Update discriminator ###
        disc_opt.zero_grad()
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device, im_dim)        
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            fake_all_feature_flatten = fake.cpu().numpy().reshape(cur_batch_size, -1)
            real_all_feature_flatten = real.cpu().numpy().reshape(cur_batch_size, -1)
            visualize_tsne(fake_all_feature_flatten)
            visualize_tsne(real_all_feature_flatten)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
