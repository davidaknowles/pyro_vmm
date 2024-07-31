from pyro import poutine
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta, AutoMultivariateNormal, init_to_value

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO, RenyiELBO, Predictive
from pyro.optim import Adam
import torch.nn.functional as F

import pymde

from sklearn.cluster import KMeans


class CSRDataset(Dataset):
    """Tiny helper class to allow sparse matrices as a torch Dataset"""

    def __init__(self, counts):
        self.counts = counts

    def __len__(self):
        return self.counts.shape[0]

    def __getitem__(self, idx):
        return self.counts[idx].todense().A1  # i hate numpy sometimes. todense() gives a np.matrix, A1 gets the array


class AnchorEmbedder:
    """Helper class for protecting new data into an existing pyMDE embedding"""

    def __init__(self, X, verbose=True):
        mde = pymde.preserve_neighbors(X, verbose=verbose)
        self.embedding = mde.embed()
        self.X = X
        self.anchor_constraint = pymde.Anchored(
            anchors=torch.arange(X.shape[0]),
            values=mde.X,
        )

    def increment(self, X_new, verbose=True, eps=1e-6):
        incremental_mde = pymde.preserve_neighbors(
            torch.cat([self.X, X_new]),
            constraint=self.anchor_constraint,
            init='random',
            verbose=verbose)

        return incremental_mde.embed(eps=eps, verbose=verbose)[self.X.shape[0]:]


@torch.no_grad()
def gmm_sample(gmm, sample_shape=torch.Size()):
    """Sample from a Gaussian Mixture Model also returning the cluster assignments."""
    sample_len = len(sample_shape)
    batch_len = len(gmm.batch_shape)
    gather_dim = sample_len + batch_len
    es = gmm.event_shape

    # mixture samples [n, B]
    mix_sample = gmm.mixture_distribution.sample(sample_shape)
    mix_shape = mix_sample.shape

    # component samples [n, B, k, E]
    comp_samples = gmm.component_distribution.sample(sample_shape)

    # Gather along the k dimension
    mix_sample_r = mix_sample.reshape(
        mix_shape + torch.Size([1] * (len(es) + 1))
    )
    mix_sample_r = mix_sample_r.repeat(
        torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
    )

    samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
    return mix_sample, samples.squeeze(gather_dim)


class Decoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, z_dim):
        super().__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, obs_dim)

    def forward(self, z):
        hidden = F.softplus(self.fc1(z))
        return self.fc21(hidden)


class Encoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, z_dim):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = F.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


class VAE(nn.Module):

    def __init__(
            self, total_N, obs_dim, hidden_dim=400, z_dim=50, K=10,
            auto_guide_type=AutoDelta, z_prior="normal", norm_count_factor=1e4,
            scale_sharing="K_z"):
        """
        z_prior can be "normal", "vampprior", "vmm" or "gmm"
        scale_sharing can be "K_z" (different for each K and z_dim, default), "z" (per z_dim), "shared" (just one)
        z_dim is the latent dimensions
        hidden_dim is the width of the encoder/decoder hidden layers
        """

        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(obs_dim, hidden_dim, z_dim)
        self.decoder = Decoder(obs_dim, hidden_dim, z_dim)
        self.z_prior = z_prior
        self.norm_count_factor = norm_count_factor
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.K = K
        self.total_N = total_N
        self.scale_sharing = scale_sharing

        # set up GMM variational approximation
        self.auto_guide = auto_guide_type(
            poutine.block(
                self.model,
                expose=["pi", "pseudo_inputs", "mix_locs", "mix_scales", "theta"]))

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        #pyro.module("encoder", self.encoder) # happening twice??
        one = torch.tensor(1., device=x.device)
        theta = pyro.sample("theta", dist.Gamma(2. * one, 0.2 * one).expand([self.obs_dim]).to_event(
            1))  # NB concentration. Gamma(shape,rate). Mean = 10.

        if self.z_prior in ["gmm", "vmm", "vampprior"]:
            # mixture proportions
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(self.K, device=x.device)))
            mix = dist.Categorical(pi)  # note this is summed over therefore not sampled

            # component centers and scales
            if self.z_prior in ["vmm", "vampprior"]:
                pseudo_inputs = pyro.sample("pseudo_inputs",
                                            dist.Normal(0 * one, one).expand([self.K, self.obs_dim]).to_event(
                                                2))  # scale 5
                mix_locs, mix_scales = self.encoder(pseudo_inputs)
            else:  # for GMM
                mix_locs = pyro.sample("mix_locs",
                                       dist.Normal(0 * one, one).expand([self.K, self.z_dim]).to_event(2))  # scale 5

            if self.z_prior in ["gmm", "vmm"]:  # for VMM this overwrites the q scale
                if self.scale_sharing == "K_z":
                    mix_scales = pyro.sample("mix_scales",
                                             dist.Gamma(10. * one, 30. * one).expand([self.K, self.z_dim]).to_event(
                                                 2))  # G(10,10): mean=1, var~1/10
                elif self.scale_sharing == "z":
                    mix_scales = pyro.sample("mix_scales",
                                             dist.Gamma(10. * one, 30. * one).expand([self.z_dim]).to_event(
                                                 1))  # G(10,10): mean=1, var~1/10
                elif self.scale_sharing == "K":
                    mix_scales = pyro.sample("mix_scales", dist.Gamma(10. * one, 30. * one).expand([self.K]).to_event(
                        1))  # G(10,10): mean=1, var~1/10
                else:
                    mix_scales = pyro.sample("mix_scales", dist.Gamma(10. * one, 30. * one))

            # construct gaussian mixture model
            comp = dist.Normal(mix_locs, mix_scales).to_event(1)
            prior_z = dist.MixtureSameFamily(mix, comp)
        else:
            prior_z = dist.Normal(
                torch.zeros([x.shape[0], self.z_dim], device=x.device),
                torch.ones([x.shape[0], self.z_dim], device=x.device)
            ).to_event(1)

        lib_size = x.sum(1)

        with pyro.plate("data", self.total_N, subsample_size=x.shape[0]):
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", prior_z)

            # decode the latent code z
            pois_mean = self.decoder(z).clamp(max=12.).exp() * lib_size[:, None]

            assert (not pois_mean.isnan().any())

            # this is a NegativeBinomial with concentration theta
            # mean=pois_mean, var = pois_mean + pois_mean^2/theta
            pyro.sample("obs", dist.GammaPoisson(theta, theta / (pois_mean + 1e-8)).to_event(1), obs=x)

        return pois_mean

    # define the guide (i.e. variational distribution) q(z|x). Same for og/GMM/VMM
    def guide(self, x):

        # don't precompute just to help with scaling to really big datasets
        norm_x = torch.log1p(self.norm_count_factor * x / x.sum(1, keepdim=True))

        self.auto_guide(x)

        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", self.total_N, subsample_size=x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            assert (not norm_x.isnan().any())
            z_loc, z_scale = self.encoder(norm_x)
            # sample the latent code z
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

    @property
    def gmm(self):
        mix_locs, mix_scales = self.encoder(self.auto_guide.pseudo_inputs)

        mix = dist.Categorical(self.auto_guide.pi)
        comp = dist.Normal(mix_locs, mix_scales).to_event(1)
        return dist.MixtureSameFamily(mix, comp)

    def gmm_sample(self, num_samples):

        return gmm_sample(self.gmm, torch.Size([num_samples]))

    def gmm_assigner(self, x):
        gmm = self.gmm
        log_prob_x = gmm.component_distribution.log_prob(gmm._pad(x))  # [S, B, k]
        log_mix_prob = torch.log_softmax(gmm.mixture_distribution.logits, dim=-1)  # [B, k]
        log_prob = log_prob_x + log_mix_prob
        probs = log_prob.softmax(-1)
        return probs, probs.argmax(-1)


def init_vae(
        X_train,
        train_loader,
        device,
        learning_rate=1e-3,
        rng_seed=42,
        norm_count_factor=1e4,
        kmeans_vmm_init=True,
        **kwargs):
    pyro.clear_param_store()

    pyro.set_rng_seed(rng_seed)

    vae = VAE(
        total_N=X_train.shape[0],
        obs_dim=X_train.shape[1],
        norm_count_factor=norm_count_factor,
        **kwargs).to(device)

    optimizer = Adam({"lr": learning_rate})

    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    for x in train_loader:  # force param init
        x = x.to(device)
        svi.evaluate_loss(x)
        break

    X_train_dense = np.asarray(X_train.todense())  # these are counts
    train_x = X_train_dense / X_train_dense.sum(1, keepdims=True)
    train_x = np.log(train_x * norm_count_factor + 1.)
    train_x_torch = torch.tensor(train_x, device=device, dtype=torch.float)

    z_loc, z_scale = vae.encoder(train_x_torch)

    store = pyro.get_param_store()

    kmeans = KMeans(n_clusters=vae.K)

    if vae.z_prior in ["vmm", "vampprior"]:
        if kmeans_vmm_init:
            centroids = kmeans._init_centroids(
                train_x,
                init='k-means++',
                x_squared_norms=None,
                sample_weight=np.ones(train_x.shape[0]),  # sklearn version dependent whether this is needed
                random_state=np.random.RandomState(seed=0))

            pseudo_inputs = torch.tensor(centroids, device=device, dtype=torch.float)

        else:
            perm = torch.randperm(train_x.size(0))
            idx = perm[:vae.K]
            pseudo_inputs = torch.tensor(train_x[idx], device=device, dtype=torch.float)

        store["AutoDelta.pseudo_inputs"] = pseudo_inputs

        mix_locs = vae.encoder(pseudo_inputs)[0]

        if vae.z_prior == "vmm":
            store["AutoDelta.mix_scales"] = torch.full_like(store["AutoDelta.mix_scales"], 0.1)

        store["AutoDelta.mix_locs"] = mix_locs

    elif vae.z_prior == "gmm":

        centroids = kmeans._init_centroids(
            z_loc.detach().cpu().numpy(),
            init='k-means++',
            x_squared_norms=None,
            sample_weight=np.ones(train_x.shape[0]),
            random_state=np.random.RandomState(seed=0))

        store = pyro.get_param_store()
        mix_locs = torch.tensor(centroids, device=device, dtype=torch.float)
        store["AutoDelta.mix_locs"] = mix_locs

        store["AutoDelta.mix_scales"] = torch.full_like(store["AutoDelta.mix_scales"], 0.1)

    return svi, vae


def one_epoch_train_or_test(svi, data_loader, device, train_flag=True):
    epoch_loss = 0.

    for x in data_loader:
        x = x.to(device)
        epoch_loss += svi.step(x) if train_flag else svi.evaluate_loss(x)

    normalizer = len(data_loader.dataset)
    total_epoch_loss = epoch_loss / normalizer
    return total_epoch_loss


def train_loop(svi, vae, train_loader, test_loader, device="cpu", num_epochs=300, test_frequency=5, save_dir=Path("."),
               file_suffix=None):
    train_elbo = []
    test_elbo = []

    if file_suffix is None:
        file_suffix = f"{vae.z_prior}_{vae.K}_{vae.scale_sharing}.pt"

    model_state_file = save_dir / f"model_state_{file_suffix}.pt"
    optim_state_file = save_dir / f"optim_state_{file_suffix}.pt"
    elbo_file = save_dir / f"elbos_{file_suffix}.pt"

    if elbo_file.is_file():
        elbos = torch.load(elbo_file)
        train_elbo = elbos["train_elbo"].tolist()
        test_elbo = elbos["test_elbo"].tolist()
        vae.load_state_dict(torch.load(model_state_file, map_location=device))
        svi.optim.load(optim_state_file)

    for epoch in range(len(train_elbo), num_epochs):
        total_epoch_loss_train = one_epoch_train_or_test(svi, train_loader, device, train_flag=True)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train), end="\r")

        if epoch % test_frequency == 0:  # report test diagnostics
            total_epoch_loss_test = one_epoch_train_or_test(svi, test_loader, device, train_flag=False)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

        svi.optim.save(optim_state_file)
        torch.save(vae.state_dict(), model_state_file)

        torch.save({"train_elbo": torch.tensor(train_elbo), "test_elbo": torch.tensor(test_elbo)}, elbo_file)

    return train_elbo, test_elbo


def estimate_marginal_likelihood(vae, x_torch, num_particles=1000):
    total_N_cache = vae.total_N

    vae.total_N = x_torch.shape[0]

    elbo = -Trace_ELBO(num_particles=num_particles).loss(vae.model, vae.guide, x_torch)
    log_px = -RenyiELBO(num_particles=num_particles).loss(vae.model, vae.guide, x_torch)

    vae.total_N = total_N_cache

    return elbo, log_px
