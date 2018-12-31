import torch
import torchvision
import PRO_GAN
import numpy as np
from tqdm import tqdm
import Losses
import landscape_dataset
import os


class ModePinningGan:

    def __init__(self):

        # select the device to be used for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(3),
            torchvision.transforms.RandomCrop((128, 128)),  # TODO: Random scale too
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(.2, .2, .1, .01),
            torchvision.transforms.ToTensor(),
          ])

        dataset = landscape_dataset.maybe_download(transforms)

        self.vgg = torchvision.models.vgg13_bn(pretrained=True).to(self.device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        num_pins = 32
        self.latent_size = 128
        batch_size = 32

        self.dataloader = torch.utils.data.DataLoader(dataset, num_pins, shuffle=True)
        self.anchor_targets = next(iter(self.dataloader)).to(self.device)

        self.anchor_latent_vectors = torch.randn(num_pins, self.latent_size, requires_grad=True, device=self.device)

        self.g = PRO_GAN.Generator(depth=6, latent_size=self.latent_size).to(self.device)
        self.d = PRO_GAN.Discriminator(6, self.latent_size).to(self.device)

        # anchor_optimizer = torch.optim.Adam(list(g.parameters()) + [anchor_latent_vectors], lr=.001)
        self.anchor_optimizer = torch.optim.Adam(list(self.g.parameters()), lr=.01)

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

        self.disc_optim = torch.optim.Adam(self.d.parameters(), lr=.01)
        self.gen_optim = torch.optim.Adam(self.g.parameters(), lr=.01)
        # TODO: WGAN_GP(use_gp=True) seems to be causing memory leak
        self.gan_loss = Losses.RelativisticAverageHingeGAN(self.d)

        self.eval_noise = torch.randn(64, self.latent_size, device=self.device)

    def extract_features(self, x):
        # x = normalize(x)  # TODO
        #print(x.min(), x.max())

        # This is an approximation of the transform that the torchvision models want.
        x = (x - .5) * .3

        for i in range(12):
            x = self.vgg.features[i](x)
        return x

    @staticmethod
    def create_grid(samples, img_file, scale_factor=1):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))), normalize=True)

    def optimize_generator_with_anchors(self):
        self.anchor_optimizer.zero_grad()

        generated = self.g(self.anchor_latent_vectors, 5, 0)
        assert self.anchor_targets.shape == generated.shape, "generated shape %s does not match target shape %s" % (str(generated.shape), str(self.anchor_targets.shape))
        loss = torch.mean(torch.abs(self.anchor_targets - generated))
        perceptual_loss = torch.mean(torch.abs(self.extract_features(self.anchor_targets) - self.extract_features(generated)))

        loss += perceptual_loss

        loss.backward()
        self.anchor_optimizer.step()
        return loss.item()

    def optimize_discriminator(self, real_samples):
        self.disc_optim.zero_grad()

        with torch.no_grad():
            noise = torch.randn(int(real_samples.shape[0]), self.latent_size, device=self.device)
            fake_samples = self.g(noise, 5, 0).detach()

        loss = self.gan_loss.dis_loss(real_samples, fake_samples, 5, 0)

        loss.backward()
        self.disc_optim.step()

        return loss.item()

    def optimize_generator(self, real_samples):
        self.gen_optim.zero_grad()

        noise = torch.randn(real_samples.shape[0], self.latent_size, device=self.device)
        fake_samples = self.g(noise, 5, 0).detach()

        loss = self.gan_loss.gen_loss(real_samples, fake_samples, 0, 5)

        loss.backward()
        self.gen_optim.step()

        return loss.item()

    def train(self, epochs=5000):
        max_mem_used = 0

        print('{"chart": "GLO Loss", "axis": "epochs"}')
        print('{"chart": "Discriminator Loss", "axis": "epochs"}')
        print('{"chart": "Generator Loss", "axis": "epochs"}')

        for epoch in tqdm(range(epochs)):

            # For the first phase, just train using the anchors. This is faster.
            if epoch < 500:
                glo_loss = self.optimize_generator_with_anchors()
                print('{"chart": "GLO Loss", "x": %d, "y": %f}' % (epoch, glo_loss))
            else:
                d_loss = 0
                g_loss = 0
                for batch in self.dataloader:
                    batch = batch.to(self.device)
                    d_loss += self.optimize_discriminator(batch)
                    g_loss += self.optimize_generator(batch)
                print('{"chart": "Discriminator Loss", "x": %d, "y": %f}' % (epoch, d_loss))
                print('{"chart": "Generator Loss", "x": %d, "y": %f}' % (epoch, g_loss))
                glo_loss = self.optimize_generator_with_anchors()
                print('{"chart": "GLO Loss", "x": %d, "y": %f}' % (epoch, glo_loss))

            with torch.no_grad():
                generated = self.g(self.eval_noise, 5, 0).detach()

                output_dir = os.path.expanduser(os.getenv('ARTIFACTS_DIR', 'samples'))
                filename = output_dir+'/%05d.jpg' % epoch
                self.create_grid(generated, filename)

            # Test for CUDA memory leaks.
            if torch.cuda.device_count() > 0 and torch.cuda.max_memory_allocated() > max_mem_used:
                max_mem_used = torch.cuda.max_memory_allocated()
                print("New CUDA allocation: " + str(max_mem_used))


    """def interpolate_latent_vectors(vector_a, vector_b):
        vectors = torch.zeros(64, latent_size, device=device)
        for i in range(64):
            vectors[i] = vector_b * (i / 64.0) + vector_a * (1 - i / 64.0)
        generated = g(vectors, 4, 0)

        filename = 'samples/interpolation.png'
        create_grid(generated, filename)
        plt.rcParams['figure.figsize'] = [10, 10]
        plt.imshow(cv2.imread(filename))
        plt.show()"""


if __name__ == "__main__":
    ModePinningGan().train()
