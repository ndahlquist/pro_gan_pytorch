import shutil
from sklearn.decomposition import PCA
import torch
import torchvision
import PRO_GAN
import numpy as np
from tqdm import tqdm
import Losses
import a_thousand_li_dataset
import os


class ModePinningGan:

    def __init__(self):

        # select the device to be used for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = a_thousand_li_dataset.maybe_download()

        self.vgg = torchvision.models.vgg13_bn(pretrained=True).to(self.device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        num_pins = 512  #len(dataset)
        self.latent_size = 128
        batch_size = 32

        self.dataloader = torch.utils.data.DataLoader(dataset, num_pins, shuffle=True)
        self.anchor_targets = next(iter(self.dataloader)).to(self.device)

        #self.anchor_latent_vectors = torch.randn(num_pins, self.latent_size, requires_grad=True, device=self.device)
        self.anchor_latent_vectors = self.choose_latent_vectors_with_pca()

        self.depth = 3
        self.g0 = PRO_GAN.Generator(depth=self.depth, latent_size=self.latent_size).to(self.device)
        self.d0 = PRO_GAN.Discriminator(self.depth, self.latent_size).to(self.device)

        self.anchor_optimizer = torch.optim.Adam(list(self.g0.parameters()) + [self.anchor_latent_vectors], lr=.01)

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

        self.d0_optim = torch.optim.Adam(self.d0.parameters(), lr=.00005)
        self.g0_optim = torch.optim.Adam(self.g0.parameters(), lr=.00005)
        self.gan_loss = Losses.WGAN_GP(self.d0, use_gp=True)

        self.eval_noise = torch.randn(64, self.latent_size, device=self.device)

    def choose_latent_vectors_with_pca(self):
        X_pca = []

        for i in range(self.anchor_targets.shape[0]):
            X_pca.append(self.anchor_targets[i].flatten().cpu().numpy())

        pca = PCA(n_components=self.latent_size).fit(X_pca)

        # Initialize latent vectors to PCA projections.
        return torch.tensor(pca.transform(X_pca), device=self.device).float()

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

    def optimize_generator_with_anchors(self, depth, alpha):
        self.anchor_optimizer.zero_grad()

        real_samples = self.__progressive_downsampling(self.anchor_targets, depth, alpha)

        generated = self.g0(self.anchor_latent_vectors, depth, alpha)
        assert generated.shape == real_samples.shape, "generated shape %s does not match target shape %s" % (str(generated.shape), str(real_samples.shape))
        loss = torch.mean(torch.abs(real_samples - generated))
        perceptual_loss = torch.mean(torch.abs(self.extract_features(real_samples) - self.extract_features(generated)))

        loss += perceptual_loss

        loss.backward()
        self.anchor_optimizer.step()
        return loss.item()

    def optimize_discriminator(self, real_samples, depth, alpha):
        self.d0_optim.zero_grad()

        real_samples = self.__progressive_downsampling(real_samples, depth, alpha)

        with torch.no_grad():
            noise = torch.randn(int(real_samples.shape[0]), self.latent_size, device=self.device)
            fake_samples = self.g0(noise, depth, alpha).detach()

        assert fake_samples.shape == real_samples.shape
        loss = self.gan_loss.dis_loss(real_samples, fake_samples, depth, alpha)

        loss.backward()
        self.d0_optim.step()

        return loss.item()

    def optimize_generator(self, real_samples, depth, alpha):
        self.g0_optim.zero_grad()

        real_samples = self.__progressive_downsampling(real_samples, depth, alpha)

        noise = torch.randn(real_samples.shape[0], self.latent_size, device=self.device)
        fake_samples = self.g0(noise, depth, alpha)

        assert fake_samples.shape == real_samples.shape
        loss = self.gan_loss.gen_loss(real_samples, fake_samples, depth, alpha)

        loss.backward()
        self.g0_optim.step()

        return loss.item()

    def __progressive_downsampling(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_downsample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),
                                                scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def save_checkpoint(self):
        checkpoints_dir = os.path.expanduser(os.getenv('ARTIFACTS_DIR', 'artifacts')) + "/checkpoints"
        os.makedirs(checkpoints_dir, exist_ok=True)
        torch.save(self.g0.state_dict(), checkpoints_dir + "/gen.pth")
        torch.save(self.d0.state_dict(), checkpoints_dir + "/disc.pth")

    def restore_checkpoint(self, checkpoints_dir):
        if not os.path.exists(checkpoints_dir + "/gen.pth") or not os.path.exists(checkpoints_dir + "/disc.pth"):
            return False

        self.g0.load_state_dict(torch.load(checkpoints_dir + "/gen.pth"))
        self.d0.load_state_dict(torch.load(checkpoints_dir + "/disc.pth"))
        return True

    def glo_pretrain(self):
        # For the first phase, just train using the anchors. This is faster.

        num_epochs_per_depth = 1000
        for epoch in tqdm(range(num_epochs_per_depth * self.depth)):
            # Training schedule.
            depth = epoch // num_epochs_per_depth
            if depth > self.depth:
                depth = self.depth
            alpha = (epoch % num_epochs_per_depth) / float(num_epochs_per_depth)

            print('{"chart": "Depth", "x": %d, "y": %.02f}' % (epoch, depth + alpha))

            glo_loss = self.optimize_generator_with_anchors(depth, alpha)
            print('{"chart": "GLO Loss", "x": %d, "y": %.04f}' % (epoch, glo_loss))

            if epoch % 5 == 0:
                with torch.no_grad():
                    # Demo both random and pinned latent vectors.
                    latent_vectors = torch.cat((self.anchor_latent_vectors[:18], self.eval_noise[:18]), 0)
                    generated = self.g0(latent_vectors, depth, alpha).detach()

                    samples_dir = os.path.expanduser(os.getenv('ARTIFACTS_DIR', 'artifacts')) + "/samples"
                    os.makedirs(samples_dir, exist_ok=True)
                    filename = samples_dir + '/%05d.%s' % (epoch, "png" if depth < 3 else "jpg")
                    self.create_grid(generated, filename)

    def train(self, start_epoch=0):
        max_mem_used = 0

        num_epochs_per_depth = 2000

        for epoch in tqdm(range(num_epochs_per_depth * self.depth)):

            # Training schedule.
            depth = epoch // num_epochs_per_depth
            alpha = (epoch % num_epochs_per_depth) / num_epochs_per_depth

            epoch += start_epoch

            print('{"chart": "Depth", "x": %d, "y": %.02f}' % (epoch, depth + alpha))

            glo_loss = self.optimize_generator_with_anchors(depth, alpha)
            print('{"chart": "GLO Loss", "x": %d, "y": %.04f}' % (epoch, glo_loss))

            d_loss = 0
            for batch in self.dataloader:
                batch = batch.to(self.device)
                d_loss += self.optimize_discriminator(batch, depth, alpha)
            print('{"chart": "Discriminator Loss", "x": %d, "y": %.04f}' % (epoch, d_loss))

            g_loss = 0
            for batch in self.dataloader:
                batch = batch.to(self.device)
                g_loss += self.optimize_generator(batch, depth, alpha)
            print('{"chart": "Generator Loss", "x": %d, "y": %.04f}' % (epoch, g_loss))

            if epoch % 20 == 0:
                with torch.no_grad():
                    # Demo both random and pinned latent vectors.
                    latent_vectors = torch.cat((self.anchor_latent_vectors[:18], self.eval_noise[:18]), 0)
                    generated = self.g0(latent_vectors, depth, alpha).detach()

                    samples_dir = os.path.expanduser(os.getenv('ARTIFACTS_DIR', 'artifacts')) + "/samples"
                    os.makedirs(samples_dir, exist_ok=True)
                    filename = samples_dir+'/%05d.%s' % (epoch, "png" if depth < 3 else "jpg")
                    self.create_grid(generated, filename)
                    shutil.copy(filename, "/persistent/")
                    self.save_checkpoint()

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
    print('{"chart": "Depth", "axis": "epochs"}')
    print('{"chart": "GLO Loss", "axis": "epochs"}')
    print('{"chart": "Discriminator Loss", "axis": "epochs"}')
    print('{"chart": "Generator Loss", "axis": "epochs"}')

    gan = ModePinningGan()
    gan.glo_pretrain()
    gan.train(3000)
