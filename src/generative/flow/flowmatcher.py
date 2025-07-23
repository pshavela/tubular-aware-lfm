import torch
import torchode

from src.generative.base import LatentGenerativeModel
from functools import partial


class LatentFlowMatcherModel(LatentGenerativeModel):
    """
    A flow matching model in the latent space with/without SPADE conditioning, based on https://arxiv.org/abs/2307.08698.

    Args:
        sigma_min: flow parameter.
        inference_steps: Number of ODE steps during sampling process.
        save_intermediate_steps: How many intermediate images should be saved during the sampling process for logging.
        atol: absolute tolerance for adaptive ODE step size solver.
        rtol: relative tolerance for adaptive ODE step size solver.
        kwargs: Arguments for the base class.
    """
    def __init__(self,
                 *,
                 sigma_min: float = 0,
                 inference_steps: int = 50,
                 save_intermediate_steps: int = 5,
                 atol: float = 1e-6,
                 rtol: float = 1e-3,
                 **kwargs):
        super().__init__(inference_steps=inference_steps,
                         save_intermediate_steps=save_intermediate_steps,
                         **kwargs)

        self.sigma_min = sigma_min
        self.atol = atol
        self.rtol = rtol

    def _rand_time(self, z_1):
        t = torch.rand(z_1.size(0)).to(z_1)
        return t

    def _forward(
        self,
        z_1,
        t,
        condition = None,
        contexts = None,
        context_coords = None,
        spacings = None,
        down_residuals = None,
        mid_residual = None,
        vessels = None,
    ):
        # prior sample batch
        z_0 = torch.randn_like(z_1).to(z_1)

        t_ = t.reshape((-1,) + (1,) * (z_1.dim() - 1))

        # construct flow at time t
        z_t = (1 - t_ * (1 - self.sigma_min)) * z_0 + t_ * z_1

        # actual velocity
        u_t = z_1 - (1 - self.sigma_min) * z_0

        predict_velocity = partial(self.model.forward,
                                   x=z_t, timesteps=t,
                                   context=contexts,
                                   context_coords=context_coords,
                                   spacings=spacings,
                                   down_block_additional_residuals=down_residuals,
                                   mid_block_additional_residual=mid_residual)

        # predicted velocity with or without conditioning
        v_t = predict_velocity(seg=self.pre_spade(condition)) if self.spade_cond else predict_velocity()

        weights = self.loss_weights(vessels)
        if weights is not None:
            loss = weights * self.loss(u_t.float(), v_t.float(), reduction='none')
            loss = loss.mean()
        else:
            loss = self.loss(u_t.float(), v_t.float())

        return loss, z_1, down_residuals, mid_residual

    def _sample_latents(
        self,
        z_0,
        conditions,
        contexts,
        context_coords,
        spacings,
        controlnet = None,
        cond_scale: float = 1.0,
        save_intermediates: bool = False
    ):
        """
        Solves the ODE using the model from t = 0 to 1. Based on
            https://torchode.readthedocs.io/en/latest/torchdiffeq/

        Returns:
            The solution (and intermediates) to the ODE using an adaptive step size solver.
        """
        input_size = z_0.shape
        batch_size = input_size[0]
        labels = conditions
        if self.spade_cond:
            conditions = self.pre_spade(conditions)

        # torchode requires flattened input
        def func(t, z):
            z_ = z.reshape(input_size)

            if controlnet:
                down_residuals, mid_residual = controlnet(x=z_,
                                                           timesteps=t,
                                                           context=contexts,
                                                           context_coords=context_coords,
                                                           controlnet_cond=labels,
                                                           conditioning_scale=cond_scale)
            else:
                down_residuals, mid_residual = None, None

            predict_velocity = partial(self.model.forward,
                                       x=z_, timesteps=t,
                                       context=contexts,
                                       context_coords=context_coords,
                                       spacings=spacings,
                                       down_block_additional_residuals=down_residuals,
                                       mid_block_additional_residual=mid_residual)

            v_t = self.cfg * (predict_velocity(seg=conditions) if self.spade_cond else predict_velocity())

            # classifier-free-guidance
            # either controlnet or SPADE
            if (1 - self.cfg) != 0.0:
                # simulate empty label map with 0s, same as in DataLoader
                empty = torch.zeros_like(labels).to(labels)

                if controlnet:
                    empty_down_residuals, empty_mid_residual = controlnet(
                        x=z_, timesteps=t,
                        contronet_cond=empty,
                        context=contexts,
                        context_coords=context_coords,
                        conditioning_scale=cond_scale
                    )
                    v_t += (1.0 - self.cfg) * self.model.forward(
                        x=z_, timesteps=t,
                        context=contexts,
                        context_coords=context_coords,
                        spacings=spacings,
                        down_block_additional_residuals=empty_down_residuals,
                        mid_block_additional_residual=empty_mid_residual
                    )
                elif self.spade_cond:
                    empty = self.pre_spade(empty)
                    v_t += (1.0 - self.cfg) * predict_velocity(seg=empty)

            return v_t.flatten(start_dim=1)

        term = torchode.ODETerm(func)
        step_method = torchode.Dopri5(term=term)
        step_size_controller = torchode.IntegralController(atol=self.atol, rtol=self.rtol, term=term)

        if save_intermediates:
            t = torch.linspace(0.0, 1.0, self.inference_steps).to(z_0).float()
            problem = torchode.InitialValueProblem(y0=z_0.flatten(start_dim=1), t_eval=t.repeat((batch_size, 1)))
        else:
            # better performance if we do not have intermediate evaluation points
            t_start = torch.tensor(batch_size * (0,)).to(z_0).float()
            t_end = torch.tensor(batch_size * (1,)).to(z_0).float()
            problem = torchode.InitialValueProblem(y0=z_0.flatten(start_dim=1), t_start=t_start, t_end=t_end)

        adjoint = torchode.AutoDiffAdjoint(step_method, step_size_controller)
        sol = adjoint.solve(problem)

        # need to reshape back to latent form,
        # shape is currently BNZ, where N = number timesteps and Z is flattend latent dimension
        z = sol.ys

        if save_intermediates:
            # move number of timesteps before batches
            z = torch.movedim(z, 1, 0)
            # conform with ddpm return structure and reshape to proper latent dimensions
            return [z[i].reshape(input_size) for i in range(z.shape[0])][::self.save_intermediate_steps]

        # Shape is BERR[R]
        return z[:, -1].reshape(input_size)
