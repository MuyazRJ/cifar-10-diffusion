# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# Exponential Moving Average (EMA) wrapper for stabilising model inference.
# EMA weights are updated during training and used at sampling time to produce
# smoother, more reliable outputs than the raw optimiser weights.
#
# Reference:
# - Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging", 1992

import torch
import copy

class EMA:
    def __init__(self, model, decay=0.9999):
        # Store a frozen copy of the model to accumulate the running average
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        # Blend current weights into the EMA: ema = decay * ema + (1 - decay) * weight
        ema_params   = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())

        for name in ema_params.keys():
            ema_params[name].data.mul_(self.decay).add_(
                model_params[name].data, alpha=(1 - self.decay)
            )

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)