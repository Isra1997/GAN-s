import torch.nn as nn
import torchvision.transforms as transforms
from Generator import Generator
from discriminator import discriminrator
import torch
import torchvision.datasets as datasets
import tqdm



# loss function
criterion = nn.BCEWithLogitsLoss()

# paper trained in 15K epochs
n_epochs=15000

# the dimension of the visual stimulus
z_dim=0

# display Step
display_step=1000

# batch size in paper is 50
batch_size= 50

# paper used a learning rate of 0.0001
lr = 0.0001

# CPU-> small datasets
# GPU-> for large datasets and ddep networks
device = 'cpu'


# noise function however used to get images to pass to the generator which we consider as noise
def load_images():
    # transform an images of shape[3,288,287] grescale and to a tensor
    tranform_X = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                   transforms.ToTensor()])

    train_set_X = datasets.ImageFolder("/Users/israragheb/Desktop/Masters_Dataset/Train",transform=tranform_X)

    # dataloader loads our data in batches can also shuffle our data
    train_loader = torch.utils.data.DataLoader(train_set_X,batch_size=batch_size,shuffle=True)

    return train_loader

# used to reprsent the real data that we wish to produce in the future from the generator
def load_Spike_Trains():

    return

gen = Generator().to(device)
gen_opt = torch.optim.Adam(gen.parameters(),lr=lr)
disc = discriminrator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(),lr=lr)

def get_disc_loss(gen,disc,criterion,real,batch_size,device):
    fake_noise = load_images(batch_size)
    fake = gen(fake_noise)
    y_pred_fake = disc(fake.detach())
    y_pred_real = disc(real)
    fake_loss = criterion(y_pred_fake, torch.zeros_like(y_pred_fake))
    real_loss = criterion(y_pred_real, torch.ones_like(y_pred_real))
    disc_loss = (fake_loss.mean() + real_loss.mean()) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    y_pred_fake = disc(fake)
    gen_loss = criterion(y_pred_fake,torch.ones_like(y_pred_fake))
    return gen_loss

def Train():
    cur_step =0
    mean_generator_loss = 0
    mean_discriminrator_loss = 0
    test_generator = True
    gen_loss = False
    error = False
    for epoch in n_epochs:
        for real,_ in tqdm(load_images()):
            cur_batch_size = len(real)

            # flattening step removed

            # ask the doctor about why we need to call the zero_grad()
            disc_opt.zero_grad()

            # Calculate the discriminrator loss
            disc_loss = get_disc_loss(gen,disc,criterion,real,batch_size,device)

            #update gradents
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

            gen_opt.zero_grad()

            gen_loss = get_gen_loss()
            





def main():
    for inputs,labels in load_images():
        print(inputs.size())
        print(labels)

if __name__ == '__main__':
    main()