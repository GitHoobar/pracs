import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
        

class Discriminator(nn.Module):
    def __init__(self, input_dim = 784, hidden_dim = 128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(nn.Linear(input_dim, hidden_dim*4),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Linear(hidden_dim*4, hidden_dim*2),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Linear(hidden_dim*2, hidden_dim),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Linear(hidden_dim, 1),
                                  nn.Sigmoid())
    def forward(self, x):
        return self.disc(x)    

class Generator(nn.Module):
    def __init__(self, zdim, img_dim, hidden_dim = 128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(zdim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim*2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim*2, hidden_dim*4),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim*4, img_dim),
                                 nn.Sigmoid())

    def forward(self, x):
        return self.gen(x)

device = torch.device('mps')
lr = 5e-5
z_dim = 64        
img_dim = 28 * 28 * 1
batch_size = 32
epochs = 50
fixed_noise = torch.randn(batch_size, z_dim).to(device)

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)

dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ###Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake)/2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()


        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        ##for tensorboard
        if batch_idx ==0:
            print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}")
            
            with torch.inference_mode():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize = True)
                img_grid_real = torchvision.utils.make_grid(data, normalize = True)

                writer_fake.add_image('Mnist Fake Images', img_grid_fake, global_step = step)
                writer_real.add_image('Mnist Real Images', img_grid_real, global_step = step)

