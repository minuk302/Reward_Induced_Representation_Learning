import src.sprites_datagen.moving_sprites as moving_sprites
import torch.utils.data.dataloader
import src.sprites_datagen.rewards as sprites_rewards
import torch.optim as optim
import torch_optimizer as optim_lookahead  # Ensure you have torch-optimizer installed
import numpy as np
import torch
import wandb

class ImageEncoder(torch.nn.Module):
    def __init__(self, input_channels, resolution, latent_dim):
        super(ImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv2d(input_channels, 4, kernel_size=(3,3), stride=(2,2), padding=(1,1)))
        for i in range(int(np.log2(resolution))-1):
            self.conv_layers.append(torch.nn.Conv2d(self.conv_layers[-1].out_channels,
                                                   self.conv_layers[-1].out_channels * 2,
                                                   kernel_size=(3,3),
                                                   stride=(2,2),
                                                   padding=(1,1)))
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(self.conv_layers[-1].out_channels, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class RewardPredictionModel(torch.nn.Module):
    def __init__(self, latent_dim, prediction_count):
        super(RewardPredictionModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.reward_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1)
        )
        self.prediction_count = prediction_count

    def forward(self, x):
        h = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        c = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)

        outputs, (h, c) = self.lstm(x, (h, c))
        predictions = []
        last_output = outputs[:, -1, :]  # Get the last output from the LSTM
        for i in range(self.prediction_count):
            outputs, (h, c) = self.lstm(last_output.unsqueeze(1), (h, c))  # Input the last output back into the LSTM
            prediction = self.reward_head(h[-1, :, :])
            predictions.append(prediction)
            last_output = outputs[:,-1,:]  # Use the prediction as the next input
        return torch.stack(predictions, dim=1)

if __name__ == '__main__':
    batch_size = 16
    trajectory_length = 50
    prior_count = 3
    reward_prediction_count = 25
    image_latent_dimension = 64
    reward_specifier = 'zero'
    resolution = 64

    wandb.init(
        project="implementation_training",
    )

    train_dataset = moving_sprites.MovingSpriteDataset(moving_sprites.AttrDict(
        resolution=resolution,
        max_seq_len=trajectory_length,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=1,      # number of shapes per trajectory
        rewards=[sprites_rewards.ZeroReward],
    ))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    
    image_encoder = ImageEncoder(3, resolution, image_latent_dimension)
    reward_prediction_model = RewardPredictionModel(image_latent_dimension, reward_prediction_count)
    
    optimizer = optim_lookahead.RAdam(
    list(image_encoder.parameters()) + list(reward_prediction_model.parameters()), 
    lr=0.001, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()

    for epoch in range(1000):
        for idx, batch in enumerate(train_loader):
            for time in range(trajectory_length):
                if time < prior_count:
                    continue
                if time > trajectory_length - reward_prediction_count:
                    continue
                images_in_window = batch['images'][:,time:time+prior_count,:,:,:]
                trajectory_image_latents = image_encoder(images_in_window.reshape(batch_size*prior_count, 3, resolution, resolution)).reshape(batch_size,prior_count,image_latent_dimension)
                estimated_rewards = reward_prediction_model(trajectory_image_latents)[:,:,-1]
                target_rewards = batch['rewards'][reward_specifier][:, time:time+reward_prediction_count]

                loss = criterion(estimated_rewards, target_rewards)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({"representaiton_model_loss": loss})
    wandb.finish()
