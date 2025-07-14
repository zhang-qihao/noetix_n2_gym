import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

DISC_LOGIT_INIT_SCALE = 1.0

class CDiscriminator(nn.Module):
    def __init__(self,
                 observation_dim,
                 observation_horizon,
                 num_motions,
                 device,
                 reward_coef=0.1,
                 reward_lerp=0.3,
                 shape=[1024, 512],
                 style_reward_function="quad_mapping",
                 **kwargs,
                 ):
        if kwargs:
            print("Discriminator.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super(CDiscriminator, self).__init__()
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.input_dim = observation_dim * observation_horizon + num_motions
        self.device = device
        self.reward_coef = reward_coef
        self.reward_lerp = reward_lerp
        self.style_reward_function = style_reward_function
        self.shape = shape

        self.softmax = nn.Softmax(dim=-1)

        discriminator_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.shape:
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)

        self.fc_dis = nn.Linear(hidden_dim, 1)
        self.fc_aux = nn.Linear(hidden_dim, num_motions)

        # for m in self.architecture.modules():
        #     if getattr(m, "bias", None) is not None:
        #         torch.nn.init.zeros_(m.bias) 

        # torch.nn.init.uniform_(self.fc_dis.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        # torch.nn.init.zeros_(self.fc_dis.bias) 

        self.train()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        hidden = self.architecture(x)

        fc_dis = self.fc_dis(hidden)
        fc_aux = self.fc_aux(hidden)

        classes = self.softmax(fc_aux)
        return fc_dis, classes
    
    def get_disc_weights(self):
        weights = []
        for m in self.architecture.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        return weights
    
    def get_disc_logit_weights(self):
        return torch.flatten(self.fc_dis.weight)
    
    def eval_disc(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        hidden = self.architecture(x)

        fc_dis = self.fc_dis(hidden)
        fc_aux = self.fc_aux(hidden)

        classes = self.softmax(fc_aux)
        return fc_dis, classes

    def compute_grad_pen(self, expert_data, skill_labels, lambda_=10):
        disc, _ = self.eval_disc(expert_data, skill_labels)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=torch.ones(disc.size(), device=disc.device), create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * grad.norm(2, dim=1).pow(2).mean()
        return grad_pen
    
    def compute_div_grad_pen(self, expert_data, expert_skill_labels, policy_data, policy_skill_labels, p=6, k=2):
        expert_d, _ = self.eval_disc(expert_data, expert_skill_labels)
        expert_grad = autograd.grad(
            outputs=expert_d, inputs=expert_data,
            grad_outputs=torch.ones(expert_d.size(), device=expert_d.device), create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        expert_grad_norm = expert_grad.view(expert_grad.size(), -1).pow(2).sum(1) ** (p / 2)

        policy_d, _ = self.eval_disc(policy_data, policy_skill_labels)
        policy_grad = autograd.grad(
            outputs=policy_d, inputs=policy_data,
            grad_outputs=torch.ones(policy_d.size(), device=policy_d.device), create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        policy_grad_norm = policy_grad.view(policy_grad.size(), -1).pow(2).sum(1) ** (p / 2)

        grad_pen = torch.mean(expert_grad_norm + policy_grad_norm) * k / 2
        return grad_pen
    
    def compute_ca_loss(self, expert_data, mismatched_labels, lambda_=1):
        d, _ = self.eval_disc(expert_data, mismatched_labels)
        loss = lambda_ * torch.nn.BCEWithLogitsLoss()(d, torch.zeros_like(d))
        return loss
    def compute_weight_decay(self, lambda_=0.0001):
        disc_weights = self.get_disc_weights()
        disc_weights = torch.cat(disc_weights, dim=-1)
        weight_decay = lambda_ * torch.sum(torch.square(disc_weights))
        return weight_decay
    
    def compute_logit_reg(self, lambda_=0.05):
        logit_weights = self.get_disc_logit_weights()
        disc_logit_loss = lambda_ * torch.sum(torch.square(logit_weights))
        return disc_logit_loss

    def predict_amp_reward(self, state_buf, skill_labels, task_reward, dt, state_normalizer=None, style_reward_normalizer=None):
        with torch.no_grad():
            self.eval()
            if state_normalizer is not None:
                for i in range(self.observation_horizon):
                    state_buf[:, i] = state_normalizer.normalize(state_buf[:, i].clone())
            dis_output, aux_output = self.eval_disc((state_buf.flatten(1, 2)), skill_labels)
            if self.style_reward_function == "quad_mapping":
                style_reward = torch.clamp(1 - (1/4) * torch.square(dis_output - 1), min=0)
            elif self.style_reward_function == "log_mapping":
                style_reward = (-torch.log(torch.maximum(1 - 1 / (1 + torch.exp(-dis_output)), torch.tensor(0.0001, device=self.device))))
            elif self.style_reward_function == "wasserstein_mapping":
                if style_reward_normalizer is not None:
                    style_reward = style_reward_normalizer.normalize(dis_output.clone())
                    style_reward_normalizer.update(dis_output)
                else:
                    style_reward = dis_output
            else:
                raise ValueError("Unexpected style reward mapping specified")
            style_reward += torch.sum(skill_labels * torch.log(torch.maximum(aux_output, torch.tensor(0.0001, device=self.device))), dim=1).unsqueeze(-1)
            style_reward *= (1.0 - self.reward_lerp) * self.reward_coef * dt
            task_reward = task_reward.unsqueeze(-1) * self.reward_lerp
            reward = style_reward + task_reward
            self.train()
        return reward.squeeze(), style_reward.squeeze()