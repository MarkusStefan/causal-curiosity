import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
import torch.nn.functional as F

class RSSMCell(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, hidden_dim, num_classes=32, class_dim=32):
        super(RSSMCell, self).__init__()
        self.state_dim = state_dim  # Stochastic latent state
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim  # Deterministic hidden state
        self.num_classes = num_classes
        self.class_dim = class_dim
        
        self.state_dim_total = self.num_classes * self.class_dim

        # Prior network (Dynamics Predictor)
        self.prior_rnn = nn.GRUCell(self.state_dim_total + self.action_dim, self.hidden_dim)
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.num_classes * self.class_dim)
        )

        # Posterior network (Encoder's latent part)
        self.posterior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.num_classes * self.class_dim)
        )

    def forward(self, prev_state, prev_action, observation_embedding):
        prev_state_flat = prev_state.view(prev_state.size(0), -1)
        
        # 1. Prior (Dynamics Predictor)
        # Predict the next hidden state
        rnn_input = torch.cat([prev_state_flat, prev_action], dim=-1)
        prior_hidden = self.prior_rnn(rnn_input, prev_state.hidden) # (Not standard GRU, but conceptually similar)

        # Predict the next latent state distribution from the hidden state
        prior_logits = self.prior_mlp(prior_hidden)
        prior_logits = prior_logits.view(-1, self.num_classes, self.class_dim)
        prior_dist = OneHotCategorical(logits=prior_logits)
        
        # 2. Posterior (Encoder's latent part)
        # Take the new observation and the predicted hidden state to get the posterior
        posterior_mlp_input = torch.cat([prior_hidden, observation_embedding], dim=-1)
        posterior_logits = self.posterior_mlp(posterior_mlp_input)
        posterior_logits = posterior_logits.view(-1, self.num_classes, self.class_dim)
        posterior_dist = OneHotCategorical(logits=posterior_logits)

        # Straight-Through Gumbel-Softmax Trick
        # The straight-through trick is a way to handle discrete sampling
        # and backpropagate gradients. We sample a one-hot vector but pass
        # the gradients through the continuous logits.
        posterior_sample = posterior_dist.sample()
        posterior_sample = posterior_sample + (posterior_logits - posterior_logits.detach())
        
        return prior_dist, posterior_dist, posterior_sample, prior_hidden
    

class Encoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=1024):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embed_dim) # Assuming 64x64 input image
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, state_dim_total, hidden_dim, out_channels=3):
        super(Decoder, self).__init__()
        self.state_dim_total = state_dim_total
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim_total + self.hidden_dim, 256 * 4 * 4),
            nn.ELU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
            # No final activation for reconstruction loss (e.g., MSE) or Sigmoid for BCE
        )

    def forward(self, state_sample, hidden_state):
        x = torch.cat([state_sample.view(state_sample.size(0), -1), hidden_state], dim=-1)
        x = self.mlp(x)
        x = x.view(-1, 256, 4, 4)
        return self.decoder(x)
    


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, hidden_dim):
        super(WorldModel, self).__init__()
        self.encoder = Encoder(in_channels=3, embed_dim=embed_dim)
        self.rssm_cell = RSSMCell(state_dim, action_dim, embed_dim, hidden_dim)
        self.decoder = Decoder(state_dim, hidden_dim, out_channels=3)
        # You would also have Reward and Discount predictors here
        
    def forward(self, observations, actions, prev_hidden_state, prev_state):
        
        # Process a sequence of observations and actions
        batch_size, seq_len, C, H, W = observations.size()
        
        all_prior_dists = []
        all_posterior_dists = []
        all_posterior_samples = []
        all_hidden_states = []

        for t in range(seq_len):
            obs_t = observations[:, t, :, :, :]
            action_t = actions[:, t, :]
            
            # Encode observation
            embed_t = self.encoder(obs_t)
            
            # Update RSSM state
            prior_dist, posterior_dist, posterior_sample, hidden_state = \
                self.rssm_cell(prev_state, action_t, embed_t)
            
            all_prior_dists.append(prior_dist)
            all_posterior_dists.append(posterior_dist)
            all_posterior_samples.append(posterior_sample)
            all_hidden_states.append(hidden_state)
            
            # Update state for next step
            prev_state = posterior_sample
            
        return all_prior_dists, all_posterior_dists, all_posterior_samples, all_hidden_states

    def get_losses(self, obs, actions, prior_dists, posterior_dists, posterior_samples, hidden_states):
        # Reconstruction loss (e.g., MSE)
        reconstructed_obs = self.decoder(torch.stack(posterior_samples, dim=1), torch.stack(hidden_states, dim=1))
        reconstruction_loss = F.mse_loss(reconstructed_obs, obs, reduction='mean')

        # KL Divergence loss
        kl_loss = 0
        for prior_dist, posterior_dist in zip(prior_dists, posterior_dists):
            kl_loss += torch.distributions.kl.kl_divergence(posterior_dist, prior_dist).mean()
        
        return reconstruction_loss, kl_loss
    


if __name__ == '__main__':
    # Example usage
    batch_size = 8
    seq_len = 10
    state_dim = 32 * 32  # Example state dimension
    action_dim = 4  # Example action dimension (e.g., discrete actions)
    embed_dim = 1024
    hidden_dim = 512

    model = WorldModel(state_dim, action_dim, embed_dim, hidden_dim)

    # Random input data
    observations = torch.randn(batch_size, seq_len, 3, 64, 64)  # Batch of images
    actions = torch.randn(batch_size, seq_len, action_dim)  # Batch of actions

    prior_dists, posterior_dists, posterior_samples, hidden_states = model(observations, actions, None, None)
    
    # Compute losses
    reconstruction_loss, kl_loss = model.get_losses(observations, actions, prior_dists, posterior_dists, posterior_samples, hidden_states)
    
    print("Reconstruction Loss:", reconstruction_loss.item())
    print("KL Divergence Loss:", kl_loss.item())