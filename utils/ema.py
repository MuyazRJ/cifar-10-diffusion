import torch
import copy

class EMA:
    def __init__(self, model, decay=0.9999):
        # Keep a deep copy of the model as the EMA model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)

        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA weights using:
        ema = decay * ema + (1 - decay) * weight
        """
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())

        for name in ema_params.keys():
            ema_params[name].data.mul_(self.decay).add_(
                model_params[name].data, alpha=(1 - self.decay)
            )

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)
