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
from src.sprites_env.envs.sprites import SpritesEnv, SpritesStateEnv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(log_dict):
    if wandb.run is None:
        wandb.init(project="implementation_training",)
    wandb.log(log_dict)

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
        image_encode = x.clone()
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
    def __init__(self, latent_dim, prediction_count, num_heads):
        super(RewardPredictionModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.reward_heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            ) for _ in range(num_heads)
        ])
        self.prediction_count = prediction_count
        self.num_heads = num_heads

    def forward(self, x):
        h = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        c = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)

        outputs, (h, c) = self.lstm(x, (h, c))
        predictions = []
        last_output = outputs[:, -1, :]  # Get the last output from the LSTM
        for i in range(self.prediction_count):
            outputs, (h, c) = self.lstm(last_output.unsqueeze(1), (h, c))  # Input the last output back into the LSTM
            head_predictions = [reward_head(h[-1, :, :]) for reward_head in self.reward_heads]
            predictions.append(torch.cat(head_predictions, dim=1))
            last_output = outputs[:,-1,:]
        return torch.stack(predictions, dim=1)

def generate_dataset_spec(params):
    rewards = []
    for specifier in params['reward_specifier']:
        if 'vertical_position' == specifier:
            rewards.append(sprites_rewards.VertPosReward)
        if 'horizontal_position' == specifier:
            rewards.append(sprites_rewards.HorPosReward)
        if 'agent_x' == specifier:
            rewards.append(sprites_rewards.AgentXReward)
        if 'agent_y' == specifier:
            rewards.append(sprites_rewards.AgentYReward)
        if 'target_x' == specifier:
            rewards.append(sprites_rewards.TargetXReward)
        if 'target_y' == specifier:
            rewards.append(sprites_rewards.TargetYReward)
    
    return moving_sprites.AttrDict(
        resolution=params['resolution'],
        max_seq_len=params['trajectory_length'],
        max_speed=params['max_speed'],      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=2,      # number of shapes per trajectory
        rewards=rewards,
    )

def generate_dataloader(params):
    dataset = moving_sprites.MovingSpriteDataset(generate_dataset_spec(params))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=16,
        drop_last=True,
    )

def create_image_decoder_representation(encoder, params, is_train, is_load):
    encoder.eval()
    train_loader = generate_dataloader(params)

    decoder = ImageDecoder().to(device)
    version_name = params['version_name']
    if is_load == True:
        decoder.load_state_dict(torch.load(f'image_decoder_{version_name}.pth'))
    if is_train == False:
        return decoder

    optimizer = optim_lookahead.RAdam(decoder.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()
    for epoch in range(1):
        for idx, batch in enumerate(train_loader):
            images_in_window = batch['images'][:, 0::4, :, :, :]
            training_length = images_in_window.shape[1]
            image_encode, _ = encoder(images_in_window.reshape(params['batch_size']*training_length, 1, params['resolution'], params['resolution']))
            estimated_image = decoder(image_encode)
            estimated_image = estimated_image.reshape(params['batch_size'],training_length,1, params['resolution'], params['resolution'])

            loss = criterion(estimated_image, images_in_window)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log({f"image_reproduce_loss": loss})

            if idx % params['training_save_index'] == 0:
                print(idx)
                torch.save(decoder.state_dict(), f'image_decoder_{version_name}.pth')
                create_image(encoder, decoder, version_name)
                if idx == params['training_finishing_index']:
                    return decoder
    return decoder

def create_model(params, is_train, is_load):
    image_encoder = ImageEncoder(1, params['resolution'], params['image_latent_dimension']).to(device)
    reward_prediction_model = RewardPredictionModel(params['image_latent_dimension'], params['reward_prediction_count'], len(params['reward_specifier'])).to(device)
    version_name = params['version_name']
    if is_load == True:
        image_encoder.load_state_dict(torch.load(f'image_encoder_{version_name}.pth'))
        reward_prediction_model.load_state_dict(torch.load(f'reward_prediction_model_{version_name}.pth'))
    if is_train == False:
        return image_encoder, reward_prediction_model
    
    optimizer = optim_lookahead.RAdam(
    list(image_encoder.parameters()) + list(reward_prediction_model.parameters()), 
    lr=0.001, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()

    train_loader = generate_dataloader(params)
    for epoch in range(1):
        for idx, batch in enumerate(train_loader):
            for time in range(0, params['trajectory_length'], params['prior_count']):
                if time < params['prior_count']:
                    continue
                if time > params['trajectory_length'] - params['reward_prediction_count']:
                    continue
                images_in_window = batch['images'][:,time-params['prior_count']:time,:,:,:]
                _, trajectory_image_latents = image_encoder(images_in_window.reshape(params['batch_size']*params['prior_count'], 1, params['resolution'], params['resolution']))
                estimated_rewards = reward_prediction_model(trajectory_image_latents.reshape(params['batch_size'],params['prior_count'],params['image_latent_dimension']))

                target_rewards = []
                for specifier in params['reward_specifier']:
                    reward_slice = batch['rewards'][specifier][:, time:time+params['reward_prediction_count']]
                    target_rewards.append(reward_slice)
                target_rewards = torch.stack(target_rewards, dim=2)

                loss = criterion(estimated_rewards, target_rewards)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                log({"representaiton_model_loss": loss})
            if idx % params['training_save_index'] == 0:
                print(idx)
                torch.save(image_encoder.state_dict(), f'image_encoder_{version_name}.pth')
                torch.save(reward_prediction_model.state_dict(), f'reward_prediction_model_{version_name}.pth')
                if idx == params['training_finishing_index']:
                    return image_encoder, reward_prediction_model
    return image_encoder, reward_prediction_model

def create_image(image_encoder, image_decoder, version_name):
    seq_generator = moving_sprites.TemplateMovingSpritesGenerator(generate_dataset_spec(params))
    traj = seq_generator.gen_trajectory()
    images = traj.images[:, None].repeat(3, axis=1).astype(np.float32)
    img = make_image_seq_strip([images[None, :]], sep_val=255.0).astype(np.uint8)
    cv2.imwrite(f"original_{version_name}.png", img[0].transpose(1, 2, 0))

    images_to_input_model = torch.from_numpy(traj.images[:, None].repeat(1, axis=1).astype(np.float32) / (255./2) - 1.0)
    images_encoded, _ = image_encoder(images_to_input_model)
    images_decoded = image_decoder(images_encoded)
    images_decoded = torch.clamp(images_decoded, min=-1.0, max=1.0)
    images_decoded = ((images_decoded + 1.0) * (255./2)).detach().numpy()
    images_decoded = images_decoded.repeat(3, axis=1).astype(np.float32)
    decoded_img = make_image_seq_strip([images_decoded[None, :]], sep_val=255.0).astype(np.uint8)
    cv2.imwrite(f"estimated_{version_name}.png", decoded_img[0].transpose(1, 2, 0))


LOG_STD_MAX = 2
LOG_STD_MIN = -5

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor(((1.0) - (-1.0)) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(((1.0) + (-1.0)) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum().unsqueeze(dim=-1)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

if __name__ == '__main__':
    # for changing 0 distractor to 1 distractor or vice versa
    # change generate_dataset_spec, shapes per traj = 3 (when training representation). currently have 3 so no need!
    # change sprites.py self.n_distractors = kwarg['n_distractors'] if kwarg else 1 part to 0.
    params = {
        'is_oracle_setup': True,
        'image_scratch': False,
        'batch_size': 32,
        'num_episodes': 100000,
        'seed': 1,
        'tau': 0.05,
        'gamma': 0.9,
        'q_lr': 1e-3,
        'actor_lr': 1e-3,
        'env_ep_length' : 50,
        'env_max_speed' : 0.1,
        'distractor': 0,
    }

    # params = {
    #     'is_oracle_setup': False,
    #     'image_scratch': False,
    #     'batch_size': 256,
    #     'num_episodes': 100000,
    #     'seed': 1,
    #     'tau': 0.05,
    #     'gamma': 0.95,
    #     'q_lr': 3e-4,
    #     'actor_lr': 3e-4,
    #     'env_ep_length' : 50,
    #     'env_max_speed' : 0.1,
    #     'distractor': 0,
    # }

    if params['distractor'] == 0:
        version_name = 'zero_distractor'
    elif params['distractor'] == 1:
        version_name = 'one_distractor'

    if params['is_oracle_setup'] == False:
        params_representation = {
            'batch_size': 256,
            'version_name': version_name,
            'trajectory_length': params['env_ep_length'],
            'max_speed': params['env_max_speed'],
            'prior_count': 3,
            'reward_prediction_count': 25,
            'image_latent_dimension': 64,
            'reward_specifier': ['agent_x','agent_y','target_x','target_y'],
            'resolution': 64,
            'training_save_index': 50,
            'training_finishing_index': 10000,
        }
        if params['image_scratch'] == True:
            image_encoder_critic = ImageEncoder(1, params_representation['resolution'], params_representation['image_latent_dimension']).to(device)
            image_encoder_actor = ImageEncoder(1, params_representation['resolution'], params_representation['image_latent_dimension']).to(device)
        else:
            image_encoder_critic, reward_prediction_model = create_model(params_representation, False, True)
            image_encoder_actor, reward_prediction_model = create_model(params_representation, False, True)
            #image_decoder = create_image_decoder_representation(image_encoder, params, True, False)
    
    #learning representation now
    data_spec = moving_sprites.AttrDict(
        resolution=64,
        max_ep_len=params['env_ep_length'],
        max_speed=params['env_max_speed'],      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True,
    )

    if params['is_oracle_setup'] == True:
        env = SpritesStateEnv(follow=True, n_distractors = params['distractor'])
        state_dim = env.observation_space.shape[0]
    else:
        env = SpritesEnv(follow=True, n_distractors = params['distractor'])
        state_dim = 128 # with CNN, we get 1x64x64 -> 4x32x32 -> ... -> 128 x 1 x 1 latent variable. I will flatten and put in.
    env.set_config(data_spec)

    action_dim = env.action_space.shape[0]
    actor = Actor(state_dim, action_dim).to(device)
    qf1 = SoftQNetwork(state_dim, action_dim).to(device)
    qf2 = SoftQNetwork(state_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(state_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(state_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    if params['is_oracle_setup'] == True:
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=params['q_lr'])
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=params['actor_lr'])
    else:
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()) + list(image_encoder_critic.parameters()), lr=params['q_lr'])
        actor_optimizer = optim.Adam(list(actor.parameters()) + list(image_encoder_actor.parameters()), lr=params['actor_lr'])

    replay_buffer = collections.deque(maxlen=1000000)

    target_entropy = -torch.prod(torch.Tensor(env.observation_space.shape)).item()
    log_alpha = torch.zeros(1).to(device)
    log_alpha.requires_grad = True
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=3e-4)

    for episode in range(params['num_episodes']):
        obs = env.reset()
        #comes in as 64x64 grayscale image. change to 1x1x64x64 for encoding.
        termination = False
        total_reward = 0
        while not termination:
            if params['is_oracle_setup'] == True:
                obs_tensor = torch.Tensor(obs).to(device)
            else:
                obs_tensor = image_encoder_actor(torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0))[0].flatten()
            action, _, _ = actor.get_action(obs_tensor)
            #action = torch.Tensor(env.action_space.sample())

            action = action.detach().cpu().numpy()
            next_obs, reward, termination, info = env.step(action)
            real_next_obs = next_obs.copy()
            replay_buffer.append((obs, real_next_obs, action, reward, termination, info))

            if len(replay_buffer) > params['batch_size']:
                if params['is_oracle_setup'] == True:
                    minibatch = random.sample(replay_buffer, params['batch_size'])
                    rb_obs, rb_next_obs, rb_actions, rb_rewards, rb_terminations, rb_infos = zip(*minibatch)
                    rb_obs_tensor = torch.Tensor(rb_obs).to(device)
                    rb_next_obs_tensor = torch.Tensor(rb_next_obs).to(device)
                    rb_actions_tensor = torch.Tensor(rb_actions).to(device)
                    rb_rewards_tensor = torch.Tensor(rb_rewards).to(device)
                    rb_terminations_tensor = torch.Tensor(rb_terminations).to(device)
                    with torch.no_grad():
                        next_state_actions, next_state_log_pis, _ = actor.get_action(rb_next_obs_tensor)
                        qf1_next_target = qf1_target(rb_next_obs_tensor, next_state_actions)
                        qf2_next_target = qf2_target(rb_next_obs_tensor, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pis
                        next_q_value = rb_rewards_tensor.flatten() + (1 - rb_terminations_tensor.flatten()) * params['gamma'] * (min_qf_next_target).view(-1)

                    qf1_a_values = qf1(rb_obs_tensor, rb_actions_tensor).view(-1)
                    qf2_a_values = qf2(rb_obs_tensor, rb_actions_tensor).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    # optimize the model
                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    pi, log_pi, _ = actor.get_action(rb_obs_tensor)
                    qf1_pi = qf1(rb_obs_tensor, pi)
                    qf2_pi = qf2(rb_obs_tensor, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(params['tau'] * param.data + (1 - params['tau']) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(params['tau'] * param.data + (1 - params['tau']) * target_param.data)

                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(rb_obs_tensor)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()
                else:
                    minibatch = random.sample(replay_buffer, params['batch_size'])
                    rb_obs, rb_next_obs, rb_actions, rb_rewards, rb_terminations, rb_infos = zip(*minibatch)

                    rb_actions_tensor = torch.Tensor(rb_actions).to(device)
                    rb_rewards_tensor = torch.Tensor(rb_rewards).to(device)
                    rb_terminations_tensor = torch.Tensor(rb_terminations).to(device)
                    rb_obs_tensor = torch.Tensor(rb_obs).to(device)
                    rb_next_obs_tensor = torch.Tensor(rb_next_obs).to(device)

                    with torch.no_grad():
                        next_state_actions, next_state_log_pis, _ = actor.get_action(image_encoder_actor(rb_next_obs_tensor.unsqueeze(1))[0].squeeze())
                        qf1_next_target = qf1_target(image_encoder_critic(rb_next_obs_tensor.unsqueeze(1))[0].squeeze(), next_state_actions)
                        qf2_next_target = qf2_target(image_encoder_critic(rb_next_obs_tensor.unsqueeze(1))[0].squeeze(), next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pis
                        next_q_value = rb_rewards_tensor.flatten() + (1 - rb_terminations_tensor.flatten()) * params['gamma'] * (min_qf_next_target).view(-1)

                    qf1_a_values = qf1(image_encoder_critic(rb_obs_tensor.unsqueeze(1))[0].squeeze(), rb_actions_tensor).view(-1)
                    qf2_a_values = qf2(image_encoder_critic(rb_obs_tensor.unsqueeze(1))[0].squeeze(), rb_actions_tensor).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    pi, log_pi, _ = actor.get_action(image_encoder_actor(rb_obs_tensor.unsqueeze(1))[0].squeeze())
                    qf1_pi = qf1(image_encoder_critic(rb_obs_tensor.unsqueeze(1))[0].squeeze(), pi)
                    qf2_pi = qf2(image_encoder_critic(rb_obs_tensor.unsqueeze(1))[0].squeeze(), pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(params['tau'] * param.data + (1 - params['tau']) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(params['tau'] * param.data + (1 - params['tau']) * target_param.data)

                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(image_encoder_actor(rb_obs_tensor.unsqueeze(1))[0].squeeze())
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                log({'qf_loss': qf_loss})
                log({'actor_loss': actor_loss})
                q_target_value_log = float(next_q_value.cpu().numpy()[0])
                log({'min_q_value': q_target_value_log})
                log({'alpha': alpha})
                log({'alpha_loss': alpha_loss})
            obs = next_obs
            total_reward += reward
            log({'reward': reward})
        log({'episodic_reward': total_reward})