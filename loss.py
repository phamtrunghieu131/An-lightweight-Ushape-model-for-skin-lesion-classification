import torch

def tversky_kahneman_loss(target, output, gamma=4/3, alpha=0.5):
  target_positive = torch.flatten(target)
  output_positive = torch.flatten(output)

  true_pos = torch.sum(target_positive * output_positive)
  true_neg = torch.sum((1-target_positive) * (1-output_positive))
  false_neg = torch.sum(target_positive * (1-output_positive))
  false_pos = torch.sum((1-target_positive) * output_positive)

  p = (alpha*true_pos + (1-alpha)*true_neg)/(alpha*true_pos + (1-alpha)*true_neg + 0.5*(false_pos + false_neg))   ###########
  p_gamma = torch.pow(p, gamma) #p^gamma
  _p_gamma = torch.pow(1-p, gamma) #(1-p)^gamma
  loss = _p_gamma/torch.pow(p_gamma + _p_gamma, 1/gamma)

  return loss