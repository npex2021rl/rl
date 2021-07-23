import numpy as np
import torch
import os


def cg(f_Ax, b, actor, obs_batch, cg_iters=10, residual_tol=1e-10):
    """
    # https://github.com/openai/baslines/blob/master/baselines.common/cg.py
    conjugate gradient algorithm
    here, f_Ax is a function which computes matrix-vector product efficiently
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p, actor, obs_batch)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    # return a search direction by solving Ax = g, where g : gradient of loss, and A : Fisher information matrix
    return x


def fisher_vector_product(v, actor, obs_batch, cg_damping=1e-2):
    # efficient Hessian-vector product
    # in our implementation, Hessian just corresponds to Fisher information matrix I
    v.detach()
    kl = torch.mean(kl_div(actor=actor, old_actor=actor, obs_batch=obs_batch))

    kl_grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = torch.cat([grad.view(-1) for grad in kl_grads])

    kl_grad_p = torch.sum(kl_grad * v)
    Iv = torch.autograd.grad(kl_grad_p, actor.parameters()) # product of Fisher information I and v
    Iv = flatten(Iv)

    return Iv + v * cg_damping


def backtracking_line_search(old_actor, actor, actor_loss, actor_loss_grad,
                             old_policy, params, maximal_step, max_kl,
                             adv, states, actions):
    backtrac_coef = 1.0
    alpha = 0.5
    beta = 0.5
    flag = False

    expected_improve = (actor_loss_grad * maximal_step).sum(0, keepdim=True)

    for i in range(10):
        new_params = params + backtrac_coef * maximal_step
        update_model(actor, new_params)

        new_actor_loss = surrogate_loss(actor, adv, states, old_policy.detach(), actions)

        loss_improve = new_actor_loss - actor_loss
        expected_improve *= backtrac_coef
        improve_condition = loss_improve / expected_improve

        kl = kl_div(actor=actor, old_actor=old_actor, obs_batch=states)
        kl = kl.mean()

        if kl < max_kl and improve_condition > alpha:
            flag = True
            break

        backtrac_coef *= beta

    if not flag:
        params = flat_params(old_actor)
        update_model(actor, params)


def kl_div(actor, old_actor, obs_batch):
    """
    Kullback-Leibler divergence between two action distributions ($\pi(\cdot \vert s ; \phi)$ and \pi(\cdot \vert s ; \phi_\text{old})$)
    we assume that both distributions are Gaussian with diagonal covariance matrices
    """
    mu, sigma = actor(torch.Tensor(obs_batch))

    mu_old, sigma_old = old_actor(torch.Tensor(obs_batch))
    mu_old = mu_old.detach()
    sigma_old = sigma_old.detach()

    kl = torch.log(sigma / sigma_old) + (sigma_old**2 + (mu_old - mu)**2) / (2.0 * sigma**2) - 0.5

    # return a batch [kl_0, ... , kl_{N-1}]^T
    # shape : [batch_size, 1]
    kl_batch = torch.sum(kl, dim=1, keepdim=True)

    return kl_batch


def flatten(hess):
    flat_hess = []
    for hessian in hess:
        flat_hess.append(hessian.contiguous().view(-1))
    flat_hess = torch.cat(flat_hess).data
    return flat_hess


def surrogate_loss(actor, adv, states, old_log_probs, actions):

    log_probs = actor.log_prob(states, actions)
    loss = torch.exp(log_probs - old_log_probs) * adv
    loss = loss.mean()

    return loss


def flat_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        len_params = len(params.view(-1))
        new_param = new_params[index: index + len_params]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += len_params
