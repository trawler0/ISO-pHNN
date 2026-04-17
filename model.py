"""Model components for port-Hamiltonian system identification."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from kan import KAN


class VTU(nn.Module):
    """
    Vector-to-upper-triangular unflattening helper.

    Parameters
    ----------
    N : int
        Matrix dimension for the N x N output.
    strict : bool, optional
        If True, uses strictly upper-triangular indices; otherwise includes diagonal.
    """

    def __init__(self, N, strict=False):
        super(VTU, self).__init__()

        idx = []
        for j in range(N):
            rng = range(j + 1, N) if strict else range(j, N)
            for i in rng:
                idx.append(j * N + i)
        self.register_buffer('idx', torch.tensor(idx, dtype=torch.long))
        self.N = N
        self.register_buffer("zeros", torch.zeros(N * N))

    def forward(self, x):
        """
        Expand a compact vector into an N x N matrix.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, L) where L matches the number of stored indices.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, N, N) with zeros outside the indexed entries.
        """
        B, L = x.shape
        assert len(self.idx) == L
        zeros = self.zeros.unsqueeze(0).expand(B, -1)
        idx = self.idx.unsqueeze(0).expand(B, -1)

        M = torch.scatter(zeros, 1, idx, x)
        M = M.view(B, self.N, self.N)
        return M



class MLP(nn.Sequential):
    """
    LayerNorm MLP with configurable depth.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer width.
    output_dim : int
        Output feature dimension.
    depth : int
        Number of hidden layers.
    batchnorm_input : bool, optional
        If True, prepends BatchNorm1d on the input.
    activation : callable, optional
        Activation constructor used in each hidden block.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            depth,
            batchnorm_input=False,
            activation=partial(nn.SiLU, inplace=True),
    ):
        layers = [nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
        )]
        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation(),
            ))
        layers.append(nn.Linear(hidden_dim, output_dim))
        if batchnorm_input:
            layers = [nn.BatchNorm1d(input_dim)] + layers
        super(MLP, self).__init__(*layers)


class Baseline(nn.Module):
    """
    Baseline model that predicts (xdot, y) = (f(x, u), g(x, u)).

    Parameters
    ----------
    input_dim : int
        State dimension.
    hidden_dim : int
        Hidden layer width.
    u_dim : int
        Input dimension.
    depth : int
        Number of hidden layers.
    """

    def __init__(self, input_dim, hidden_dim, u_dim, depth):
        super(Baseline, self).__init__()
        self.mlp = MLP(input_dim + u_dim, hidden_dim, input_dim + u_dim, depth=depth)

    def forward(self, x, u):
        """
        Compute predicted state derivative and output.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).
        u : torch.Tensor
            Input tensor of shape (batch, u_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (xdot, y) with shapes (batch, input_dim) and (batch, u_dim).
        """
        u_dim = u.shape[-1]
        xu = torch.cat([x, u], dim=-1)
        out = self.mlp(xu)
        return out[:, :-u_dim], out[:, -u_dim:]
    
    def get_params(self):
        return self.parameters()



class JLinear(nn.Module):
    """
    Learnable skew-symmetric J using a dense parameter matrix.

    Parameters
    ----------
    dim : int
        State dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.J_ = nn.Parameter(torch.randn(dim, dim))

        nn.init.kaiming_normal_(self.J_)

    def forward(self, x, grad_H):
        """
        Apply the skew-symmetric matrix to grad_H.

        Parameters
        ----------
        x : torch.Tensor
            State tensor (unused).
        grad_H : torch.Tensor
            Gradient of Hamiltonian with shape (batch, dim).

        Returns
        -------
        torch.Tensor
            J(x) * grad_H with shape (batch, dim).
        """
        J = self.J_ - self.J_.T
        return F.linear(grad_H, J)

    def reparam(self, x):
        """
        Return the full J matrix for each batch element.

        Parameters
        ----------
        x : torch.Tensor
            State tensor used for batch sizing only.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, dim, dim).
        """
        J = (self.J_ - self.J_.T)
        return J.unsqueeze(0).expand(x.size(0), -1, -1).permute(0, 2, 1)


class RLinear(nn.Module):
    """
    Learnable positive semi-definite R via R_ R_^T.

    Parameters
    ----------
    dim : int
        State dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.R_ = nn.Parameter(torch.randn(dim, dim))

        nn.init.kaiming_normal_(self.R_)

    def forward(self, x, grad_H):
        """
        Apply the dissipative matrix to grad_H.

        Parameters
        ----------
        x : torch.Tensor
            State tensor (unused).
        grad_H : torch.Tensor
            Gradient of Hamiltonian with shape (batch, dim).

        Returns
        -------
        torch.Tensor
            R(x) * grad_H with shape (batch, dim).
        """
        return F.linear(F.linear(grad_H, self.R_), self.R_.T) / math.sqrt(self.R_.size(0))

    def reparam(self, x):
        """
        Return the full R matrix for each batch element.

        Parameters
        ----------
        x : torch.Tensor
            State tensor used for batch sizing only.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, dim, dim).
        """
        R = self.R_ @ self.R_.T / math.sqrt(self.R_.size(0))
        return R.unsqueeze(0).expand(x.size(0), -1, -1)


# G linear in the sense of corresponding to linear PH-system, but it means it is constant
class GLinear(nn.Module):
    """
    Constant input matrix G.

    Parameters
    ----------
    dim : int
        Output dimension of G.
    """

    def __init__(self, dim):
        super().__init__()
        self.G_ = nn.Parameter(torch.randn(1, dim))

    def forward(self, x):
        """
        Return the constant G expanded to the batch size.

        Parameters
        ----------
        x : torch.Tensor
            State tensor used for batch sizing only.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, dim).
        """
        return self.G_.expand(x.size(0), -1)


# quadratic Hamiltonian
class Grad_HLinear(nn.Module):
    """
    Quadratic Hamiltonian gradient: grad_H(x) = Q Q^T x + bias.

    Parameters
    ----------
    dim : int
        State dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.Q_ = nn.Parameter(torch.randn(dim, dim))
        self.bias = nn.Parameter(torch.randn(dim))

        nn.init.kaiming_normal_(self.Q_)


    def forward(self, x):
        """
        Compute grad_H for a quadratic Hamiltonian.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, dim).

        Returns
        -------
        torch.Tensor
            Gradient of H with shape (batch, dim).
        """
        Q = self.Q_ @ self.Q_.T
        return F.linear(x, Q, bias=self.bias)

    def H(self, x):
        """
        Evaluate the quadratic Hamiltonian.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, dim).

        Returns
        -------
        torch.Tensor
            Scalar Hamiltonian values with shape (batch,).
        """
        Q = self.Q_ @ self.Q_.T
        return (F.linear(x, Q, bias=self.bias) @ x.unsqueeze(-1)).squeeze(-1)


class JDefault(nn.Module):
    """
    State-dependent skew-symmetric J parameterized by an MLP or KAN.

    Parameters
    ----------
    input_dim : int
        State dimension.
    hidden_dim : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    arch : {"mlp", "kan"}, optional
        Backbone network type.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim,
            depth,
            arch="mlp"
    ):
        super().__init__()
        if arch == "mlp":
            self.J_ = MLP(input_dim, hidden_dim, ((input_dim - 1) * input_dim) // 2, depth)
        elif arch == "kan":
            self.J_ = KAN([input_dim] + [hidden_dim] * depth + [((input_dim - 1) * input_dim) // 2])
        self.vsu = VTU(input_dim, strict=True)

    def forward(self, x, grad_H):
        """
        Apply J(x) to grad_H.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).
        grad_H : torch.Tensor
            Gradient of Hamiltonian with shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            J(x) * grad_H with shape (batch, input_dim).
        """
        J_ = self.vsu(self.J_(x))
        J = J_ - J_.permute(0, 2, 1)
        return (grad_H.unsqueeze(1) @ J).squeeze(1)

    def reparam(self, x):
        """
        Return the full J matrix for each batch element.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, input_dim, input_dim).
        """
        J_ = self.vsu(self.J_(x))
        J = J_ - J_.permute(0, 2, 1)
        return J.permute(0, 2, 1)


class RDefault(nn.Module):
    """
    State-dependent dissipative matrix R parameterized by an MLP or KAN.

    Parameters
    ----------
    input_dim : int
        State dimension.
    hidden_dim : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    arch : {"mlp", "kan"}, optional
        Backbone network type.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim,
            depth,
            arch="mlp"
    ):
        super().__init__()
        if arch == "mlp":
            self.R_ = MLP(input_dim, hidden_dim, input_dim ** 2, depth)
        elif arch == "kan":
            self.R_ = KAN([input_dim] + [hidden_dim] * depth + [input_dim ** 2])

    def forward(self, x, grad_H):
        """
        Apply R(x) to grad_H.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).
        grad_H : torch.Tensor
            Gradient of Hamiltonian with shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            R(x) * grad_H with shape (batch, input_dim).
        """
        R_ = self.R_(x)
        d = int(math.sqrt(R_.size(-1)))
        R_ = R_.view(R_.size(0), d, d)
        return (grad_H.unsqueeze(1) @ R_ @ R_.permute(0, 2, 1)).squeeze(1) / math.sqrt(d)

    def reparam(self, x):
        """
        Return the full R matrix for each batch element.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, input_dim, input_dim).
        """
        R_ = self.R_(x)
        d = int(math.sqrt(R_.size(-1)))
        R_ = R_.view(R_.size(0), d, d)
        R = R_ @ R_.permute(0, 2, 1) / math.sqrt(d)
        return R

class HuberActivation(nn.Module):
    """Absolute-value activation used to enforce nonnegativity."""

    def forward(self, x):
        return F.huber_loss(x, torch.zeros_like(x), reduction="none")

class Grad_H_positive(nn.Module):
    """
    Hamiltonian gradient model with nonnegative energy via Abs activation.

    Parameters
    ----------
    input_dim : int
        State dimension.
    hidden_dim : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    arch : {"mlp", "kan"}, optional
        Backbone network type.
    """

    def __init__(self, input_dim, hidden_dim, depth, arch="mlp"):
        super().__init__()
        if arch == "mlp":
            self.H = nn.Sequential(MLP(input_dim, hidden_dim, 1, depth), HuberActivation())
        elif arch == "kan":
            self.H = nn.Sequential(KAN([input_dim] + [hidden_dim] * depth + [1]), HuberActivation())

    def forward(self, x):
        """
        Compute grad_H(x) using autograd.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Gradient of H with shape (batch, input_dim).
        """
        with torch.enable_grad():
            x.requires_grad = True
            H = self.H(x)
            grad_H = torch.autograd.grad(
                inputs=x,
                outputs=H,
                grad_outputs=torch.ones_like(H),
                create_graph=True,
                retain_graph=True,
            )[0]
        return grad_H
    
class Grad_H(nn.Module):
    """
    Hamiltonian gradient model.

    Parameters
    ----------
    input_dim : int
        State dimension.
    hidden_dim : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    arch : {"mlp", "kan"}, optional
        Backbone network type.
    """

    def __init__(self, input_dim, hidden_dim, depth, arch="mlp"):
        super().__init__()
        if arch == "mlp":
            self.H = MLP(input_dim, hidden_dim, 1, depth)
        elif arch == "kan":
            self.H = KAN([input_dim] + [hidden_dim] * depth + [1])

    def forward(self, x):
        """
        Compute grad_H(x) using autograd.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Gradient of H with shape (batch, input_dim).
        """
        with torch.enable_grad():
            x.requires_grad = True
            H = self.H(x)
            grad_H = torch.autograd.grad(
                inputs=x,
                outputs=H,
                grad_outputs=torch.ones_like(H, device=x.device),
                create_graph=True,
                retain_graph=True,
            )[0]
        return grad_H

# just for the spring
class RQuadratic(nn.Module):
    """
    Quadratic parametrization of R for spring systems.

    Parameters
    ----------
    D : int
        State dimension.
    """

    def __init__(self, D):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(D+1, D**2))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x, grad_H):
        """
        Apply quadratic R(x) to grad_H.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, D).
        grad_H : torch.Tensor
            Gradient of Hamiltonian with shape (batch, D).

        Returns
        -------
        torch.Tensor
            R(x) * grad_H with shape (batch, D).
        """
        B, D = x.size()
        x = torch.cat([x, torch.ones((B, 1), device=x.device)], dim=1)
        x = (x @ self.weight).view(-1, D, D)
        R = x @ x.transpose(1, 2) / math.sqrt(D)
        return (grad_H.unsqueeze(1) @ R).squeeze(1)

    def reparam(self, x):
        """
        Return the full R matrix for each batch element.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, D).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, D, D).
        """
        B, D = x.size()
        x = torch.cat([x, torch.ones((B, 1), device=x.device)], dim=1)
        x = (x @ self.weight).view(-1, D, D)
        R = x @ x.transpose(1, 2) / math.sqrt(D)
        return R

# one can often guess J from calculations with Poisson brackets
class JSpring(nn.Module):
    """
    Fixed J matrix for the spring system.
    """

    def __init__(self):
        super().__init__()
        J = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=torch.float32)
        self.register_buffer("J", J)

    def forward(self, x, grad_H):
        """
        Apply the fixed J matrix to grad_H.

        Parameters
        ----------
        x : torch.Tensor
            State tensor (unused).
        grad_H : torch.Tensor
            Gradient of Hamiltonian with shape (batch, 4).

        Returns
        -------
        torch.Tensor
            J * grad_H with shape (batch, 4).
        """
        return F.linear(grad_H, self.J.T)

    def reparam(self, x):
        """
        Return the fixed J matrix.

        Parameters
        ----------
        x : torch.Tensor
            State tensor (unused).

        Returns
        -------
        torch.Tensor
            Tensor of shape (4, 4).
        """
        return self.J

class J_Sigmoid(nn.Module):
    """
    Skew-symmetric J constructed from sigmoid-gated factors.

    Parameters
    ----------
    input_dim : int
        State dimension.
    hidden_dim : int
        Hidden layer width.
    excitation : {"linear", "mlp"}, optional
        Form of the sigmoid excitation network.
    """

    def __init__(self, input_dim, hidden_dim, excitation="linear"):
        super().__init__()
        self.input_dim = input_dim
        self.J1 = nn.Parameter(torch.randn(input_dim, input_dim + hidden_dim).T)
        self.J2 = nn.Parameter(torch.randn(input_dim + hidden_dim, input_dim).T)
        if excitation == "linear":
            self.J_sigma = nn.Linear(input_dim, hidden_dim)
        elif excitation == "mlp":
            self.J_sigma = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )

        nn.init.kaiming_normal_(self.J1)
        nn.init.kaiming_normal_(self.J2)

    def forward(self, x, grad_H):
        """
        Apply J(x) to grad_H using sigmoid gating.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).
        grad_H : torch.Tensor
            Gradient of Hamiltonian with shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            J(x) * grad_H with shape (batch, input_dim).
        """
        sigma = torch.sigmoid(self.J_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)
        J_left = F.linear((F.linear(grad_H, self.J1) * sigma), self.J2)
        J_right = F.linear((F.linear(grad_H, self.J2.T) * sigma), self.J1.T)
        return J_left - J_right

    def reparam(self, x):
        """
        Return the full J matrix for each batch element.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, input_dim, input_dim).
        """
        sigma = torch.sigmoid(self.J_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)  # B x hidden
        sigma = torch.diag_embed(sigma)  # B x hidden x hidden
        J1 = self.J1.T.unsqueeze(0).expand(x.size(0), -1, -1)  # B x input x hidden
        J2 = self.J2.T.unsqueeze(0).expand(x.size(0), -1, -1)
        J_left = J1 @ sigma @ J2  # B x input x input
        J_right = J2.permute(0, 2, 1) @ sigma @ J1.permute(0, 2, 1)  # B x input x input
        return J_left - J_right

class R_Sigmoid(nn.Module):
    """
    Dissipative R constructed from sigmoid-gated factors.

    Parameters
    ----------
    input_dim : int
        State dimension.
    hidden_dim : int
        Hidden layer width.
    excitation : {"linear", "mlp"}, optional
        Form of the sigmoid excitation network.
    """

    def __init__(self, input_dim, hidden_dim, excitation="linear"):
        super().__init__()
        self.input_dim = input_dim
        self.R_ = nn.Parameter(torch.randn(input_dim, input_dim + hidden_dim).T)
        if excitation == "linear":
            self.R_sigma = nn.Linear(input_dim, hidden_dim)
        elif excitation == "mlp":
            self.R_sigma = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )

        nn.init.kaiming_normal_(self.R_)

    def forward(self, x, grad_H):
        """
        Apply R(x) to grad_H using sigmoid gating.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).
        grad_H : torch.Tensor
            Gradient of Hamiltonian with shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            R(x) * grad_H with shape (batch, input_dim).
        """
        sigma = torch.sigmoid(self.R_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)
        x = F.linear(F.linear(grad_H, self.R_) * sigma, self.R_.T)
        return x

    def reparam(self, x):
        """
        Return the full R matrix for each batch element.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, input_dim, input_dim).
        """
        sigma = torch.sigmoid(self.R_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)
        sigma = torch.diag_embed(sigma)
        R_ = self.R_.T.unsqueeze(0).expand(x.size(0), -1, -1)
        R = R_ @ sigma @ R_.permute(0, 2, 1)
        return R


class PHNNModel(nn.Module):
    """
    Port-Hamiltonian Neural Network model.

    Parameters
    ----------
    input_dim : int
        State dimension.
    hidden_dim : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    J : str, optional
        J parametrization: {"linear", "default", "default_kan", "sigmoid", "spring"}.
    R : str, optional
        R parametrization: {"linear", "quadratic", "default", "sigmoid", "default_kan"}.
    grad_H : str, optional
        Hamiltonian gradient parametrization: {"gradient", "gradient_positive",
        "gradient_kan", "linear"}.
    G : str, optional
        Input matrix parametrization: {"mlp", "kan", "linear"}.
    excitation : str, optional
        Excitation type for sigmoid-based parametrizations.
    u_dim : int, optional
        Input dimension.
    """

    def __init__(self, input_dim, hidden_dim, depth, J="default", R="default", grad_H="gradient", G="mlp", excitation="mlp",
                 u_dim=1):
        super().__init__()
        if J == "linear":
            self.J = JLinear(input_dim)
        elif J == "default":
            self.J = JDefault(input_dim, hidden_dim, depth)
        elif J == "default_kan":
            self.J = JDefault(input_dim, hidden_dim, depth, arch="kan")
        elif J == "sigmoid":
            self.J = J_Sigmoid(input_dim, hidden_dim, excitation)
        elif J == "spring":
            self.J = JSpring()
        else:
            raise ValueError("Unknown J")

        if R == "linear":
            self.R = RLinear(input_dim)
        elif R == "quadratic":
            self.R = RQuadratic(input_dim)
        elif R == "default":
            self.R = RDefault(input_dim, hidden_dim, depth)
        elif R == "sigmoid":
            self.R = R_Sigmoid(input_dim, hidden_dim, excitation)
        elif R == "default_kan":
            self.R = RDefault(input_dim, hidden_dim, depth, arch="kan")
        else:
            raise ValueError("Unknown R")

        if grad_H == "gradient":
            self.grad_H = Grad_H(input_dim, hidden_dim, depth)
        elif grad_H == "gradient_kan":
            self.grad_H = Grad_H(input_dim, hidden_dim, depth, arch="kan")
        elif grad_H == "gradient_positive":
            self.grad_H = Grad_H_positive(input_dim, hidden_dim, depth)
        elif grad_H == "gradient_kan_positive":
            self.grad_H = Grad_H_positive(input_dim, hidden_dim, depth, arch="kan")
        elif grad_H == "linear":
            self.grad_H = Grad_HLinear(input_dim)
        else:
            raise ValueError("Unknown grad_H")

        if G == "mlp":
            self.G = MLP(input_dim, hidden_dim, input_dim * u_dim, depth)
        elif G == "kan":
            self.G = KAN([input_dim] + [hidden_dim] * depth + [input_dim * u_dim])
        elif G == "linear":
            self.G = GLinear(input_dim * u_dim)
        else:
            raise ValueError("Unknown G")
        
    def get_params(self):
        if isinstance(self.J, JLinear) or isinstance(self.J, JSpring):
            J_params = {"params": self.J.parameters(), "weight_decay": 0}
        else:
            J_params = {"params": self.J.parameters()}
        
        if isinstance(self.R, RLinear) or isinstance(self.R, RQuadratic):
            R_params = {"params": self.R.parameters(), "weight_decay": 0}
        else:
            R_params = {"params": self.R.parameters()}

        if isinstance(self.grad_H, Grad_HLinear):
            H_params = {"params": self.grad_H.parameters(), "weight_decay": 0}
        else:
            H_params = {"params": self.grad_H.parameters()}

        if isinstance(self.G, GLinear):
            G_params = {"params": self.G.parameters(), "weight_decay": 0}
        else:
            G_params = {"params": self.G.parameters()}

        params = [J_params, R_params, H_params, G_params]
        return params
            

    def forward(self, x, u):
        """
        Compute the port-Hamiltonian dynamics and output.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).
        u : torch.Tensor
            Input tensor of shape (batch, u_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (xdot, y) with shapes (batch, input_dim) and (batch, u_dim).
        """
        grad_H = self.grad_H(x)
        J_hat = self.J(x, grad_H)
        R_hat = self.R(x, grad_H)
        G = self.G(x)

        G = G.view(G.size(0), G.size(1) // u.size(-1), u.size(-1))

        y = (G.transpose(1, 2) @ grad_H.unsqueeze(-1)).squeeze(-1)
        input_term = (G @ u.unsqueeze(-1)).squeeze(-1)

        xdot = J_hat - R_hat + input_term

        return xdot, y

    def reparam(self, x):
        """
        Return J and R matrices for each batch element.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (J, R) each with shape (batch, input_dim, input_dim).
        """
        J = self.J.reparam(x)
        R = self.R.reparam(x)
        return J, R

    def H(self, x):
        """
        Evaluate the Hamiltonian for the given state.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Scalar Hamiltonian values with shape (batch,).
        """
        return self.grad_H.H(x)


if __name__ == "__main__":
    model = PHNNModel(16, 16, 3, u_dim=8, J="spring_chain", R="spring_chain", grad_H="spring_chain", G="spring_chain")
    x = torch.randn(256, 16)
    u = torch.randn(256, 8)
    print(sum(p.numel() for p in model.parameters()))
    print(model(x, u)[0].shape, model(x, u)[1].shape)
