import numpy as np
import torch
import torch.nn.functional as F

class StaticCRF:
    """
    A static CRF implementation that performs mean-field inference as post-processing, optimised for binary case.
    """
    def __init__(self, n_spatial_dims, filter_size=11, n_iter=5,
            returns='logits', smoothness_weight=1, smoothness_theta=1):
        self.n_spatial_dims = n_spatial_dims
        self.n_iter = n_iter
        self.filter_size = np.broadcast_to(filter_size, n_spatial_dims)
        self.returns = returns
        self.smoothness_weight = smoothness_weight
        inv_theta = 1 / np.broadcast_to(smoothness_theta, n_spatial_dims)
        self.inv_smoothness_theta = torch.tensor(inv_theta, dtype=torch.float)

    def __call__(self, x, spatial_spacings=None):
        """
        Apply binary CRF post-processing to input logits
        
        Args:
            x: tensor (batch_size, 1, *spatial) with logits (positive = foreground)
            spatial_spacings: array (batch_size, n_spatial_dims), defaults to ones
        Returns:
            Refined logits or probabilities based on 'returns' setting
        """

        batch_size, channels, *spatial = x.shape
        assert len(spatial) == self.n_spatial_dims
        assert channels == 1, "Input must be single-channel for binary segmentation"

        # Move parameters to input device
        self.inv_smoothness_theta = self.inv_smoothness_theta.to(x.device)

        # Default spatial spacings
        if spatial_spacings is None:
            spatial_spacings = np.ones((batch_size, self.n_spatial_dims))

        # Convert single channel to 2 classes
        # For BCE, positive values = foreground probability > 0.5
        x = torch.cat([x, -x], dim=1)  # [pos, neg] logits

        unary = x.clone()

        # Mean-field inference iterations
        for _ in range(self.n_iter):
            # Convert to probabilities
            x = torch.sigmoid(x[:, 0:1])
            x = torch.cat([x, 1 - x], dim=1)

            # Message passing: apply the smoothing filter
            x = self.smoothness_weight * self._smoothing_filter(x, spatial_spacings)
            # Compatibility transform (can use a fixed function)
            x = self._compatibility_transform(x)
            # Combine with the original unaries
            x = unary - x

        if self.returns == 'logits':
            output = x
        elif self.returns == 'proba':
            output = F.softmax(x, dim=1)
        elif self.returns == 'log-proba':
            output = F.log_softmax(x, dim=1)
        else:
            raise ValueError("Attribute `returns` must be 'logits', 'proba' or 'log-proba'.")
        
        # Convert back to one channel
        output = output[:, 0] - output[:, 1] if self.returns == 'logits' else output[:, 0]
        output = output.unsqueeze(1)

        return output

    def _smoothing_filter(self, x, spatial_spacings):
        # Apply the smoothing filter for each example.
        return torch.stack([self._single_smoothing_filter(x[i], spatial_spacings[i]) for i in range(x.shape[0])])

    @staticmethod
    def _pad(x, filter_size):
        padding = []
        for fs in filter_size:
            padding += 2 * [fs // 2]
        return F.pad(x, list(reversed(padding)))

    def _single_smoothing_filter(self, x, spatial_spacing):
        x = self._pad(x, self.filter_size)
        for i, dim in enumerate(range(1, x.ndim)):
            x = x.transpose(dim, -1)
            shape_before_flatten = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)
            # Create and apply a 1D Gaussian kernel.
            kernel = self._create_gaussian_kernel1d(self.inv_smoothness_theta[i], spatial_spacing[i],
                                                    self.filter_size[i]).view(1, 1, -1).to(x)
            x = F.conv1d(x, kernel)
            x = x.squeeze(1).view(*shape_before_flatten, x.shape[-1]).transpose(-1, dim)
        return x

    @staticmethod
    def _create_gaussian_kernel1d(inverse_theta, spacing, filter_size):
        distances = spacing * torch.arange(-(filter_size // 2), filter_size // 2 + 1).to(inverse_theta)
        kernel = torch.exp(-(distances * inverse_theta) ** 2 / 2)
        # Zero out the center (if desired), following the original code.
        zero_center = torch.ones(filter_size).to(kernel)
        zero_center[filter_size // 2] = 0
        return kernel * zero_center

    def _compatibility_transform(self, x):
        labels = torch.arange(x.shape[1], device=x.device)
        # A simple compatibility function: assign -1 where labels are equal.
        compatibility_matrix = (-(labels.unsqueeze(1) == labels.unsqueeze(0)).float()).to(x)
        return torch.einsum('ij...,jk->ik...', x, compatibility_matrix)
