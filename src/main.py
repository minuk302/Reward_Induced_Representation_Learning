import src.sprites_datagen.moving_sprites as moving_sprites
import torch.utils.data.dataloader
import src.sprites_datagen.rewards as sprites_rewards
import torch.optim as optim
import torch_optimizer as optim_lookahead  # Ensure you have torch-optimizer installed
import numpy as np
import torch
import cv2
import wandb
from src.general_utils import make_image_seq_strip

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
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = torch.nn.functional.relu(layer(x))
        image_encode = x.detach().clone()
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return image_encode, x

class ImageDecoder(torch.nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            # Input: (128, 1, 1)
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (64, 2, 2)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 4, 4)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, 8, 8)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # Output: (8, 16, 16)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),    # Output: (4, 32, 32)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),    # Output: (1, 64, 64)
        )

    def forward(self, x):
        return self.decoder(x)

class RewardPredictionModel(torch.nn.Module):
    def __init__(self, latent_dim, prediction_count):
        super(RewardPredictionModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.reward_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
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

def generate_dataset_spec(params):
    return moving_sprites.AttrDict(
        resolution=params['resolution'],
        max_seq_len=params['trajectory_length'],
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=1,      # number of shapes per trajectory
        rewards=[sprites_rewards.VertPosReward],
    )

def generate_dataloader(params):
    dataset = moving_sprites.MovingSpriteDataset(generate_dataset_spec(params))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

def create_image_decoder_representation(encoder, params, is_train, is_load, wandb):
    encoder.eval()
    train_loader = generate_dataloader(params)

    decoder = ImageDecoder()
    if is_load == True:
        decoder.load_state_dict(torch.load(f'image_decoder.pth'))
    if is_train == False:
        return decoder

    optimizer = optim_lookahead.RAdam(decoder.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()
    for epoch in range(1):
        for idx, batch in enumerate(train_loader):
            for time in range(params['trajectory_length']):
                if time < params['prior_count']:
                    continue
                if time > params['trajectory_length'] - params['reward_prediction_count']:
                    continue
                images_in_window = batch['images'][:,time-params['prior_count']:time,:,:,:]
                image_encode, _ = encoder(images_in_window.reshape(params['batch_size']*params['prior_count'], 1, params['resolution'], params['resolution']))
                estimated_image = decoder(image_encode)
                estimated_image = estimated_image.reshape(params['batch_size'],params['prior_count'],1, params['resolution'], params['resolution'])

                loss = criterion(estimated_image, images_in_window)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({f"image_reproduce_loss": loss})

            if idx % 20 == 0:
                torch.save(decoder.state_dict(), f'image_decoder.pth')
    return decoder

def create_model(params, is_train, is_load, wandb):
    image_encoder = ImageEncoder(1, params['resolution'], params['image_latent_dimension'])
    reward_prediction_model = RewardPredictionModel(params['image_latent_dimension'], params['reward_prediction_count'])
    if is_load == True:
        image_encoder.load_state_dict(torch.load('image_encoder.pth'))
        reward_prediction_model.load_state_dict(torch.load('reward_prediction_model.pth'))
    if is_train == False:
        return image_encoder, reward_prediction_model
    
    optimizer = optim_lookahead.RAdam(
    list(image_encoder.parameters()) + list(reward_prediction_model.parameters()), 
    lr=0.001, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()

    train_loader = generate_dataloader(params)
    for epoch in range(1):
        for idx, batch in enumerate(train_loader):
            for time in range(params['trajectory_length']):
                if time < params['prior_count']:
                    continue
                if time > params['trajectory_length'] - params['reward_prediction_count']:
                    continue
                images_in_window = batch['images'][:,time-params['prior_count']:time,:,:,:]
                _, trajectory_image_latents = image_encoder(images_in_window.reshape(params['batch_size']*params['prior_count'], 1, params['resolution'], params['resolution']))
                estimated_rewards = reward_prediction_model(trajectory_image_latents.reshape(params['batch_size'],params['prior_count'],params['image_latent_dimension']))[:,:,-1]
                target_rewards = batch['rewards'][params['reward_specifier']][:, time:time+params['reward_prediction_count']]

                loss = criterion(estimated_rewards, target_rewards)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({"representaiton_model_loss": loss})
            if idx % 20 == 0:
                print(idx)
                torch.save(image_encoder.state_dict(), f'image_encoder.pth')
                torch.save(reward_prediction_model.state_dict(), f'reward_prediction_model.pth')
    return image_encoder, reward_prediction_model

if __name__ == '__main__':
    # wandb.init(
    #    project="implementation_training",
    # )
    params = {
        'batch_size': 32,
        'trajectory_length': 30,
        'prior_count': 3,
        'reward_prediction_count': 10,
        'image_latent_dimension': 64,
        'reward_specifier': 'vertical_position',
        'resolution': 64,
    }
    image_encoder, reward_prediction_model = create_model(params, False, True, wandb)
    image_decoder = create_image_decoder_representation(image_encoder, params, False, True, wandb)

    seq_generator = moving_sprites.TemplateMovingSpritesGenerator(generate_dataset_spec(params))
    traj = seq_generator.gen_trajectory()
    images = traj.images[:, None].repeat(3, axis=1).astype(np.float32)
    img = make_image_seq_strip([images[None, :]], sep_val=255.0).astype(np.uint8)
    cv2.imwrite("original.png", img[0].transpose(1, 2, 0))

    images_to_input_model = torch.from_numpy(traj.images[:, None].repeat(1, axis=1).astype(np.float32) / (255./2) - 1.0)
    images_encoded, _ = image_encoder(images_to_input_model)
    images_decoded = image_decoder(images_encoded)
    images_decoded = torch.clamp(images_decoded, min=-1.0, max=1.0)
    images_decoded = ((images_decoded + 1.0) * (255./2)).detach().numpy()
    images_decoded = images_decoded.repeat(3, axis=1).astype(np.float32)
    decoded_img = make_image_seq_strip([images_decoded[None, :]], sep_val=255.0).astype(np.uint8)
    cv2.imwrite("estimated.png", decoded_img[0].transpose(1, 2, 0))