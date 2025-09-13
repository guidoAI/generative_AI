import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_training_data():
    # (re)load the MNIST data:
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
        #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
        #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

    # For auto-encoder, change the targets in the training data:
    training_x = torch.FloatTensor(len(training_data), 28, 28)
    training_y = torch.FloatTensor(len(training_data), 28, 28)
    training_labels = np.zeros((len(training_data), 1))
    for i in range(len(training_data)):
        img, label = training_data[i]
        training_x[i] = img
        training_y[i] = img
        training_labels[i] = label
    training_x = training_x.reshape((len(training_data), 1, 28, 28))
    training_y = training_y.reshape((len(training_data), 1, 28, 28))

    test_x = torch.FloatTensor(len(test_data), 28, 28)
    test_y = torch.FloatTensor(len(test_data), 28, 28)
    test_labels = np.zeros((len(training_data), 1))
    for i in range(len(test_data)):
        img, label = test_data[i]
        test_x[i] = img
        test_y[i] = img
        test_labels[i] = label
    test_x = test_x.reshape((len(test_data), 1, 28, 28))
    test_y = test_y.reshape((len(test_data), 1, 28, 28))

    return training_data, test_data, training_x, training_y, training_labels, test_x, test_y, test_labels

# function to show an image and its reconstruction:
def show_img_and_reconstr(img, reconstr):
    # show the image
    plt.figure()
    # show two subplots:
    plt.subplot(1,2,1)
    img = img.detach().cpu().numpy()
    img = img.reshape((28,28))
    plt.imshow(1-img,  cmap='gray')
    plt.title('Original')
    plt.subplot(1,2,2)
    reconstr = reconstr.detach().cpu().numpy()
    reconstr = reconstr.reshape((28,28))
    plt.imshow(1-reconstr,  cmap='gray')
    plt.title('Reconstructed')
    plt.show()

# class of fully convolutional network:
class FCNN(nn.Module):
    def __init__(self, n_hidden = 25):
        super(FCNN, self).__init__()
        self.flatten = nn.Flatten()
        
        in_channel = 1
        out_channel = 16
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding ='same') # bias = False
        self.act1 = nn.ReLU(inplace= True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear2 = nn.Linear(3136,n_hidden)
        # self.act2 = nn.ReLU(inplace= True)
        # self.linear3 = nn.Linear(512, 100)
        # self.act3 = nn.ReLU(inplace= True)
        # self.linear4 = nn.Linear(100, 512)
        # self.act4 = nn.ReLU(inplace= True)
        self.linear5 = nn.Linear(n_hidden, 2304)
        self.act5 = nn.ReLU(inplace= True)
        self.resh = nn.Unflatten(1, (16, 12, 12))
        self.deconv = nn.ConvTranspose2d(out_channel, in_channel, kernel_size=6, stride = 2, padding_mode ='zeros')
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        # convolutional pipeline:
        self.feature_maps = self.conv1(x)
        tmp = self.act1(self.feature_maps)
        tmp = self.maxpool1(tmp)
        tmp = torch.flatten(tmp, 1, -1)
        tmp = self.linear2(tmp)
        # tmp = self.act2(tmp)
        self.latent = tmp
        # tmp = self.linear3(tmp)
        # self.latent = self.act3(tmp)
        # # deconvolutional pipeline:
        # tmp = self.linear4(self.latent)
        # tmp = self.act4(tmp)
        tmp = self.linear5(tmp)
        tmp = self.act5(tmp)
        tmp = torch.flatten(tmp, 1, -1)
        tmp = self.resh(tmp)
        logits = self.deconv(tmp)
        logits = self.sigmoid(logits)

        return logits

    def decoder(self, latent):

        tmp = self.linear5(latent)
        tmp = self.act5(tmp)
        tmp = torch.flatten(tmp, 1, -1)
        tmp = self.resh(tmp)
        logits = self.deconv(tmp)
        logits = self.sigmoid(logits)

        return logits

# Train and test loop for the auto-encoder:
def train_loop(training_x, training_y, batch_size, model, loss_fn, optimizer):
    
    size = len(training_x)
    
    # shuffle the indices to get random batches:
    inds = np.random.permutation(size)
    
    num_batches = size // batch_size

    for batch in range(num_batches):
        inds_batch = inds[batch*batch_size:(batch+1)*batch_size]
        X = training_x[inds_batch]
        y = training_y[inds_batch]
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(text_x, test_y, model, loss_fn):
    size = len(test_x)
    test_loss = 0

    with torch.no_grad():
        for i in range(size):
            X = test_x[i]
            X = X.reshape((1,1,28,28))
            pred = model(X)
            Y = test_y[i]
            Y = Y.reshape((1,1,28,28))
            indiv_loss = loss_fn(pred, Y).item()
            test_loss += indiv_loss

    test_loss /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def train(n_hidden = 25, learning_rate = 1e-1, batch_size = 16, epochs = 5, model = None):
    # Create an FCNN:
    if model is None:
        model = FCNN(n_hidden=n_hidden)

    # loss function and optimizer:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(training_x, training_y, batch_size, model, loss_fn, optimizer)
        test_loop(test_x, test_y, model, loss_fn)
    # save the model weights:
    torch.save(model.state_dict(), "fcnn_model.pth")

    return model

def analyze(model):
    model.load_state_dict(torch.load("fcnn_model.pth"))

    # show a number of random samples with their reconstructions:
    n_samples = 5
    for sam in range(n_samples):
        sample_idx = torch.randint(len(training_x), size=(1,)).item()
        img = training_x[sample_idx]
        img = torch.reshape(img, (1,1,28,28))
        reconstr = model(img)
        loss = loss_fn(img, reconstr).item()
        print(f'Loss = {loss}')
        show_img_and_reconstr(img, reconstr)

    # show a number of random samples in the latent space:
    n_samples = 9
    plt.figure()
    plt.title('Random Samples from Latent Space')
    for sam in range(n_samples):
        latent = torch.randn((1,n_hidden))
        reconstr = model.decoder(latent)
        plt.subplot(3,3,sam+1)
        plt.imshow(1-reconstr.detach().cpu().numpy().reshape((28,28)),  cmap='gray')
        plt.axis('off')
    plt.show()

    # show a number of perturbations for each sample
    n_perturbations = 5
    for sam in range(n_samples):
        sample_idx = torch.randint(len(training_x), size=(1,)).item()
        img = training_x[sample_idx]
        img = torch.reshape(img, (1,1,28,28))
        reconstr_GT = model(img)
        plt.figure()
        plt.title('Perturbed reconstructions')
        plt.subplot(1,n_perturbations+1,1)
        plt.imshow(1-reconstr_GT.detach().cpu().numpy().reshape((28,28)),  cmap='gray')
        plt.axis('off')
        for i in range(n_perturbations):
            perturbation = torch.randn_like(latent) * 1.0
            latent = model.latent + perturbation
            reconstr = model.decoder(latent)
            plt.subplot(1,n_perturbations+1,i+2)
            plt.imshow(1-reconstr.detach().cpu().numpy().reshape((28,28)),  cmap='gray')
            plt.axis('off')
        plt.show()

    print('Running on the whole data set.')
    # run the model on all training samples, storing the latent space activations:
    step = 5
    latents = torch.zeros((len(training_x)//step, n_hidden))
    labels = np.zeros((len(training_x)//step,))
    with torch.no_grad():
        for i in range(0,len(training_x),step):
            X = training_x[i]
            X = X.reshape((1,1,28,28))
            _ = model(X)
            latents[i//step] = model.latent
            labels[i//step] = training_labels[i]

    # run t-SNE on the latent space:

    # Move to CPU and convert to numpy
    latent_np = latents.detach().cpu().numpy()

    print('Performing t-SNE')
    # Run t-SNE (reduce to 2D for visualization)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latent_2d = tsne.fit_transform(latent_np)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="tab10")
    plt.colorbar()
    plt.show()