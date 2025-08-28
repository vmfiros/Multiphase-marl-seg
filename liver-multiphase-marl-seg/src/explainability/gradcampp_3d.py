
import torch, torch.nn.functional as F

def grad_cam_pp_3d(model, x, target_layer_name='up1', class_idx=0):
    """Compute Grad-CAM++ on a target decoder block.
    Returns CAM volume normalized to [0,1] with shape (D,H,W).
    """
    # Hook to grab features & gradients
    feats = {}
    def fwd_hook(_, __, output): feats['act'] = output.detach()
    def bwd_hook(_, grad_in, grad_out): feats['grad'] = grad_out[0].detach()

    target_layer = getattr(model, target_layer_name)
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    p,_ = model(x)       # forward
    logit = p[:,0].mean()  # simple scalar target (mean tumor logit)
    logit.backward(retain_graph=True)

    acts = feats['act']           # (B,C,D,H,W)
    grads = feats['grad']         # (B,C,D,H,W)
    B,C,D,H,W = acts.shape

    # Grad-CAM++ weights (simplified)
    grads2 = grads**2
    grads3 = grads**3
    eps = 1e-8
    alpha = grads2 / (2*grads2 + (acts*grads3).sum(dim=(2,3,4), keepdim=True) + eps)
    weights = (alpha * F.relu(grads)).sum(dim=(2,3,4), keepdim=True)  # (B,C,1,1,1)

    cam = (weights * acts).sum(dim=1)  # (B,D,H,W)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    h1.remove(); h2.remove()
    return cam.squeeze(0).detach().cpu().numpy()
