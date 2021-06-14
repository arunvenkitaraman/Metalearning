import math
import time
import random
import torch # v0.4.1
from torch import nn
from torch.nn import functional as F
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
nh=32
kshot=4
n_meta_train=50000
reg=1e-2
n_tasks=8
# no of hidden layer neurons
def net(x, params):
    x = F.linear(x, params[0], params[1])
    x = F.relu(x)

    x = F.linear(x, params[2], params[3])
    x = F.relu(x)

    x = F.linear(x, params[4], params[5])
    return x

params = [
    torch.Tensor(nh, 1).uniform_(-1., 1.).requires_grad_(),
    torch.Tensor(nh).zero_().requires_grad_(),

    torch.Tensor(nh, nh).uniform_(-1./math.sqrt(nh), 1./math.sqrt(nh)).requires_grad_(),
    torch.Tensor(nh).zero_().requires_grad_(),

    torch.Tensor(1, nh).uniform_(-1./math.sqrt(nh), 1./math.sqrt(nh)).requires_grad_(),
    torch.Tensor(1).zero_().requires_grad_(),


]
#W = [(torch.cat([torch.eye(1*len(params[i].view(-1))),-1e-2*torch.eye(len(params[i].view(-1)))],dim=-1)) for i in range(len(params))]

W=[]
for i in range(len(params)):
    s=len(params[i].view(-1))
    w = torch.Tensor(s,2*s).uniform_(-1/math.sqrt(s*2*s),1/math.sqrt(s*2*s)).requires_grad_()
    W.append(w)
#
W = [(torch.cat([1*torch.eye(1*len(params[i].view(-1))),-1e-2*torch.eye(len(params[i].view(-1)))],dim=-1)).requires_grad_() for i in range(len(params))]
#print(W)
#print(W)
#full_params=torch.cat(params)
#pummy=torch.tensor([])
#par=list(pummy)
#for i in range(len(params)):
#
#params_net=torch.stack([params,W])

opt = torch.optim.SGD([
                {'params': params,},
                {'params': W, 'lr': 1e-4}
            ], lr=1e-3)

n_inner_loop = 1

x=[]
y=[]
v_x=[]
v_y=[]
tic=time.time()
for tasks in range(n_tasks):
    b = 0 if random.choice([True, False]) else math.pi
    tempx=torch.rand(kshot,1) * 2* math.pi
    bet=torch.rand(1)
    tempy=bet*torch.sin(tempx + b)
    x.append(tempx)
    y.append(tempy)

    #bet = torch.rand(1)
    v_tempx = torch.rand(kshot, 1) * 2 * math.pi
    v_tempy = bet*torch.sin(tempx + b)
    v_x.append(v_tempx)
    v_y.append(v_tempy)

#print(x)
#print(x.size())
for it in range(n_meta_train):

    #b = 0 if random.choice([True, False]) else math.pi
    # x = torch.rand(4, 1)*4*math.pi - 2*math.pi
    # y = torch.sin(x + b)
    #
    # v_x = torch.rand(4, 1)*4*math.pi - 2*math.pi
    # v_y = torch.sin(v_x + b)

    opt.zero_grad()

    new_params = params
    #print(W[1][0:5])
    loss=0
    for k in range(n_inner_loop):
        for tasks in range(n_tasks):
            tempx=x[tasks]
            #print(tempx)
            tempy=y[tasks]
            f = net(tempx, new_params)
            loss = loss+F.l1_loss(f, tempy)

        # create_graph=True because computing grads here is part of the forward pass.
        # We want to differentiate through the SGD update steps and get higher order
        # derivatives in the backward pass.
        grads = torch.autograd.grad(loss, new_params, create_graph=True)
        #print(grads[0])
        #print(len(torch.cat([new_params[2].view(), g
        #
        #
        #
        # rads[2].view()])))
        #    print(W[1])
    #new_params_vec = [(torch.matmul(W[i], torch.cat([params[i], grads[i]]).view(-1))) for i in range(len(params))]
    #new_params_vec = [(torch.matmul(W[i], torch.cat([params[i], torch.norm(params[i])*grads[i]]).view(-1))) for i in range(len(params))]
    new_params_vec = [(torch.matmul(W[i], torch.cat([params[i],(grads[i])]).view(-1))) for i in range(len(params))]
    new_params = [(new_params_vec[i].reshape(params[i].size())) for i in range(len(params))]
    # print('New_params',new_params)
    # print('Params',params)

    if it % 1000 == 0: print('Iteration %d -- Inner loop %d -- Loss: %.4f' % (it, k, loss/n_tasks))

    loss2=0
    #print(v_x[0:2])
    for tasks in range(n_tasks):
        v_tempx = v_x[tasks]
        v_tempy=v_y[tasks]
        v_tempf = net(v_tempx, new_params)
        loss2 = loss2 + F.l1_loss(v_tempf, v_tempy)

    loss2=loss2
    loss_tr=loss2
    loss_W=0
    for i in range(len(W)):

            loss_W=loss_W+reg*torch.norm(W[i].view(-1),p=2)**2
    loss2=loss2+loss_W
    #opt.zero_grad()

    loss2.backward()
    #print('Gradient: ', W[1].grad)
    opt.step()

    if it % 1000 == 0: print ('Iteration %d -- Outer Loss: %.4f' % (it, loss_tr/n_tasks))

    #print(*alpha,sep=',')
    #
toc=time.time()
toc=tic-toc
print('Time elapsed: %.4f' % (toc))
t_loss=0
n_tasks=100
for it in range(n_tasks):
    t_b = 0 if random.choice([True, False]) else math.pi
    #t_b = math.pi #0

    bet = torch.rand(1)
    t_x = torch.rand(kshot, 1)*2*math.pi
    t_y = bet*torch.sin(t_x + t_b)



    t_params = params
    for k in range(n_inner_loop):
        t_f = net(t_x, t_params)
        t_loss = t_loss+F.l1_loss(t_f, t_y)

        grads = torch.autograd.grad(t_loss, t_params, create_graph=True)


    #t_new_params_vec = [(torch.matmul(W[i], torch.cat([t_params[i], grads[i]]).view(-1))) for i in range(len(params))]
    #t_new_params_vec = [(torch.matmul(W[i], torch.cat([t_params[i], torch.norm(new_params[i])*grads[i]]).view(-1))) for i in range(len(params))]
    t_new_params_vec = [(torch.matmul(W[i], torch.cat([t_params[i],(grads[i])]).view(-1)))
                        for i in range(len(params))]
    t_new_params = [(t_new_params_vec[i].reshape(t_params[i].size())) for i in range(len(params))]

    #bet = torch.rand(1)
    test_x = torch.arange(0*math.pi, 2*math.pi, step=0.01).unsqueeze(1)
    test_y = bet*torch.sin(test_x + t_b)
    test_f = net(test_x, t_new_params)




    plt.plot(test_x.data.numpy(), test_y.data.numpy(), label='sin(x)')
    plt.plot(test_x.data.numpy(), test_f.data.numpy(), label='net(x)')
    plt.plot(t_x.data.numpy(), t_y.data.numpy(), 'o', label='Examples')
    plt.legend()
    plt.savefig('krmaml-full-sine.png')
    #plt.plot(W[0].data.numpy())
    #plt.savefig('W-krmaml-full.png')
    dis=t_loss/n_tasks

print('Test Loss: %.4f' % (dis))
