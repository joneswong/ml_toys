import numpy as np
import torch
from torch import optim

only_inputs=True

A = torch.as_tensor(np.identity(3, dtype=np.float32))
w = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

adapted_w = w + .0
print(adapted_w.requires_grad)

optimizer = optim.SGD([w], lr=0.5)

def loss_func(t):
    return 0.5 * torch.sum(t * torch.sum(A * t, axis=-1))

# inner loop (update on support set)
for i in range(1):
    loss = loss_func(adapted_w)#0.5 * torch.sum(adapted_w * torch.sum(A * adapted_w, axis=-1))
    # if `create_graph=False`, the calculated gradients are not a function of \theta^{(t)}
    # then \theta^{(t+1)} = \theta^{(t)} + learning_rate * grad_as_a_constant_vector
    # which blocks the BP w.r.t. \theta^{(0)}
    grads = torch.autograd.grad(loss, adapted_w, only_inputs=only_inputs, create_graph=True)
    #h = torch.autograd.functional.hessian(loss_func, adapted_w)
    print("at {}-th shot".format(i))
    print("length of returned grads: {}".format(len(grads)))
    print("explicitly calculated grad: {}".format(grads[0]))
    #print("explicitly calculated hessian: {}".format(h))
    print("grad attribute of \\w: {} with `only_inputs={}`".format(w.grad, only_inputs))
    adapted_w = adapted_w - 0.5 * grads[0]
    print("adapted \\w: {}".format(adapted_w))

print(adapted_w.requires_grad)
# update on query set
optimizer.zero_grad()
loss = loss_func(adapted_w)
loss.backward()
optimizer.step()
#grads = torch.autograd.grad(loss, w, only_inputs=False)
#print(grads)
print("finally")
print("grad attribute of \\adapted_w: {}".format(adapted_w.grad))
# to check correctness (i.e., consider just one adaptation or say one-shot), g_{maml} = g_1 - \alpha H_0 g_1
print("grad attribute of \\w: {}".format(w.grad))
print("updated \\adapted_w: {}".format(adapted_w))
print("updated \\w: {}".format(w))


