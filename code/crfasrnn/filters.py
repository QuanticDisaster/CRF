"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

try:
    import permuto_cpp
except ImportError as e:
    raise (e, "Did you import `torch` first?")

_CPU = torch.device("cpu")
_EPS = np.finfo("float").eps


class PermutoFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_in, features):
        import pdb; pdb.set_trace()
        q_out = permuto_cpp.forward(q_in, features)[0]
        ctx.save_for_backward(features)
        return q_out

    @staticmethod
    def backward(ctx, grad_q_out):
        feature_saved = ctx.saved_tensors[0]
        grad_q_back = permuto_cpp.backward(
            grad_q_out.contiguous(), feature_saved.contiguous()
        )[0]
        return grad_q_back, None  # No need of grads w.r.t. features



def _spatial_features(points, sigma):
    """
    Return the spatial features as a Tensor. The "spatial" features are probably just points coordinates XYZ

    Args:
        points:  point cloud as a Tensor of shape (n, 3)
        sigma:  Bandwidth parameter

    Returns:
        Tensor of shape (n, 3)
    """
    return points / sigma

class AbstractFilter(ABC):
    """
    Super-class for permutohedral-based Gaussian filters
    """

    def __init__(self, points):
        self.features = self._calc_features(points)
        self.norm = self._calc_norm(points)

    def apply(self, input_):
        output = PermutoFunction.apply(input_, self.features)
        return output * self.norm

    @abstractmethod
    def _calc_features(self, points):
        pass

    def _calc_norm(self, points):
        all_ones = torch.ones((len(points),1), dtype=torch.float32, device=_CPU)
        norm = PermutoFunction.apply(all_ones, self.features)
        return 1.0 / (norm + _EPS)


class SpatialFilter(AbstractFilter):
    """
    Gaussian filter in the spatial ([x, y, z]) domain
    """

    def __init__(self, points, gamma):
        """
        Create new instance

        Args:
            points:  point cloud Tensor of shape (n, 3)
            gamma:  Standard deviation
        """
        self.gamma = gamma
        super(SpatialFilter, self).__init__(points)

    def _calc_features(self, points):
        return _spatial_features(points, self.gamma)


class BilateralFilter(AbstractFilter):
    """
    Gaussian filter in the bilateral ([r, g, b, x, y]) domain
    """

    def __init__(self, points, alpha, beta):
        """
        Create new instance

        Args:
            points:  point cloud Tensor of shape (n, 3 + c)
            alpha:  Smoothness (spatial) sigma
            beta:   Appearance (features) sigma
        """
        self.alpha = alpha
        self.beta = beta
        super(BilateralFilter, self).__init__(points)

    def _calc_features(self, points):
        """xy = _spatial_features(
            points, self.alpha
        )  # TODO Possible optimisation, was calculated in the spatial kernel
        
        feats = (points[:,3:] / float(self.beta))  
        return torch.cat([xy, rgb], dim=2)
        """
        return points
