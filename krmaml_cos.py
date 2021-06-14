import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Krmaml_cos(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Krmaml_cos, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test



        self.net = Learner(config, args.imgc, args.imgsz)
        self.reg=args.reg_cos
        self.W= []
        for param in self.net.parameters():
            s = len(param.view(-1))
            w=nn.Parameter(torch.FloatTensor(s,2*s).requires_grad_())
            torch.nn.init.normal_(w)
            self.W.append(w)


        self.meta_optim = optim.Adam([{'params': self.net.parameters()},{'params': self.W, 'lr': 1e-3}], lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry,iter_num):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            #lz = list(map(lambda p: torch.stack([p[1], p[0]]).view(-1),  zip(grad, self.net.parameters()))) # almost twice slower than MAML
            lz=[(torch.stack([grad[i].view(-1),self.net.parameters()[i].view(-1)]).view(-1).to(device=device)) for i in range(len(grad))]
           
            z = [(lz[i]/(1e-7 + torch.norm(lz[i], p=2))) for i in range(len(grad))]
            
            #fast_weights = list(map(lambda p: torch.reshape(torch.matmul(p[1],  p[0]),list(p[2].size())), zip(z, self.W,grad)))
            fast_weights = [(torch.reshape(torch.matmul(self.W[i].to(device=device),  z[i]),list(grad[i].size()))) for i in range(len(grad))]
            
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            #with torch.no_grad():
                # [setsz, nway]
            logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry[i])
            losses_q[1] += loss_q
                # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry[i]).sum().item()
            corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                #lz = list(map(lambda p: torch.stack([p[1], p[0]]).view(-1),  zip(grad, fast_weights))) # almost twice slower than MAML
                #z = list(map(lambda p: p[0]/(1e-7 + torch.norm(p[0], p=2)),  zip(lz)))

                #fast_weights = list(map(lambda p: torch.reshape(torch.matmul(p[1],  p[0]),list(p[2].size())), zip(z, self.W,grad)))
                lz=[(torch.stack([grad[i].view(-1),fast_weights[i].view(-1)]).view(-1).to(device=device)) for i in range(len(grad))]
           
                z = [(lz[i]/(1e-7 + torch.norm(lz[i], p=2))) for i in range(len(grad))]
            
                fast_weights = [(torch.reshape(torch.matmul(self.W[i].to(device=device),  z[i]),list(grad[i].size()))) for i in range(len(grad))]
            
                
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        for ww in self.W:
          loss_q=loss_q+task_num*self.reg*torch.norm(ww,2)**2

        #if iter_num%10==0: 
        #  for p in self.W:
        #     p.requires_grad = True
        #  for p in self.net.parameters():
        #     p.requires_grad = True
        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects[1]) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        #lz = list(map(lambda p: torch.stack([p[1], p[0]]).view(-1),  zip(grad, net.parameters()))) # almost twice slower than MAML
        #z = list(map(lambda p: p[0]/(1e-7 + torch.norm(p[0], p=2)),  zip(lz)))

        #fast_weights = list(map(lambda p: torch.reshape(torch.matmul(p[1],  p[0]),list(p[2].size())), zip(z, self.W,grad)))
        lz=[(torch.stack([grad[i].view(-1),net.parameters()[i].view(-1)]).view(-1).to(device=device)) for i in range(len(grad))]
           
        z = [(lz[i]/(1e-7 + torch.norm(lz[i], p=2))) for i in range(len(grad))]
        fast_weights = [(torch.reshape(torch.matmul(self.W[i].to(device=device),  z[i]),list(grad[i].size()))) for i in range(len(grad))]
            
        
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            #lz = list(map(lambda p: torch.stack([p[1], p[0]]).view(-1),  zip(grad, fast_weights))) # almost twice slower than MAML
            #z = list(map(lambda p: p[0]/(1e-7 + torch.norm(p[0], p=2)),  zip(lz)))
            #fast_weights = list(map(lambda p: torch.reshape(torch.matmul(p[1],  p[0]),list(p[2].size())), zip(z, self.W,grad)))
            lz=[(torch.stack([grad[i].view(-1),fast_weights[i].view(-1)]).view(-1).to(device=device)) for i in range(len(grad))]
           
            z = [(lz[i]/(1e-7 + torch.norm(lz[i], p=2))) for i in range(len(grad))]
            
            fast_weights = [(torch.reshape(torch.matmul(self.W[i].to(device=device),  z[i]),list(grad[i].size()))) for i in range(len(grad))]
            
                
            
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects[1]) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
