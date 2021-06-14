import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse
import matplotlib.pyplot as plt

from    maml import Maml
from    metasgd import Metasgd
from    krmaml import Krmaml
from    krmaml2 import Krmaml2
from krmaml_cos import Krmaml_cos

fig_id='24Feb'
#fig_id.to(device)
def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    dim=8
    kersz=3
    config = [
        ('conv2d', [dim, 1, kersz, kersz, 2, 0]),
        ('relu', [True]),
        ('bn', [dim]),
        ('conv2d', [dim, dim, kersz, kersz, 2, 0]),
        ('relu', [True]),
        ('bn', [dim]),
        ('conv2d', [dim, dim, kersz, kersz, 2, 0]),
        ('relu', [True]),
        ('bn', [dim]),
        ('conv2d', [dim, dim, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [dim]),
        ('flatten', []),
        ('linear', [args.n_way, dim])
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maml = Maml(args, config).to(device)
    metasgd = Metasgd(args, config).to(device)
    krmaml = Krmaml(args, config).to(device)
    #krmaml2 = Krmaml2(args, config).to(device)
    krmaml_cos = Krmaml_cos(args, config).to(device)
    

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    #print(maml)
    #print('Total trainable tensors:', num)
    

    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)
   # db = OmniglotNShot('omniglot', batchsz=32, n_way=5, k_shot=1, k_query=1, imgsz=w)
    NMSE_test_maml=[]
    NMSE_test_krmaml=[]
    NMSE_test_krmaml2=[]
    NMSE_test_metasgd=[]
    NMSE_test_krmaml_cos=[]
    NMSE_train_maml=[]
    NMSE_train_krmaml=[]
    NMSE_train_krmaml2=[]
    NMSE_train_metasgd=[]
    NMSE_train_krmaml_cos=[]
    Xaxis=[]
    Xaxis2=[]

    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs_maml = maml(x_spt, y_spt, x_qry, y_qry,)
        accs_metasgd = metasgd(x_spt, y_spt, x_qry, y_qry)
        accs_krmaml = krmaml(x_spt, y_spt, x_qry, y_qry,step)
        #accs_krmaml2 = krmaml2(x_spt, y_spt, x_qry, y_qry)
        accs_krmaml_cos = krmaml_cos(x_spt, y_spt, x_qry, y_qry,step)

        if step % 50 == 0:
            torch.save({
            'maml_state_dict': maml.state_dict(),
            'metasgd_state_dict': metasgd.state_dict(),
            'krmaml_state_dict': krmaml.state_dict(),
            #'krmaml2_state_dict': krmaml2.state_dict(),
            'krmaml_cos_state_dict': krmaml_cos.state_dict(),
            'step' : step,
            'NMSE_test_maml': NMSE_test_maml,
            'NMSE_test_krmaml': NMSE_test_krmaml,
            #'NMSE_test_krmaml2': NMSE_test_krmaml2,
            'NMSE_test_metasgd': NMSE_test_metasgd,
            'NMSE_test_krmaml_cos': NMSE_test_krmaml_cos,
            'NMSE_train_maml': NMSE_train_maml,
            'NMSE_train_krmaml': NMSE_train_krmaml,
            #'NMSE_train_krmaml2': NMSE_train_krmaml2,
            'NMSE_train_metasgd': NMSE_train_metasgd,
            'NMSE_train_krmaml_cos': NMSE_train_krmaml_cos,
            'Xaxis': Xaxis,
            'Xaxis2': Xaxis2,
            }, '/content/drive/My Drive/Colab Notebooks/omniglot/'+fig_id+'.pt')

            print('step:', step, '\ttraining acc maml:', accs_maml)
            print('step:', step, '\ttraining acc metasgd:', accs_metasgd)
            print('step:', step, '\ttraining acc krmaml:', accs_krmaml)
            #print('step:', step, '\ttraining acc krmaml2:', accs_krmaml2)
            print('step:', step, '\ttraining acc krmaml_cos:', accs_krmaml_cos)

            NMSE_train_maml.append(accs_maml)
            NMSE_train_krmaml.append(accs_krmaml)
            #NMSE_train_krmaml2.append(accs_krmaml2)
            NMSE_train_metasgd.append(accs_metasgd)
            NMSE_train_krmaml_cos.append(accs_krmaml_cos)
            Xaxis.append(step)
           


            plt.clf()
            plt.plot(Xaxis,NMSE_train_maml, label='maml')
            plt.plot(Xaxis,NMSE_train_krmaml, label='krmaml-gaussian-500')
            #plt.plot(Xaxis,NMSE_train_krmaml2, label='krmaml-gaussian-5000')
            plt.plot(Xaxis,NMSE_train_krmaml_cos, label='krmaml-cosine')
            plt.plot(Xaxis,NMSE_train_metasgd, label='metasgd')
            plt.legend()
    
            plt.legend()
            fig_name='/content/drive/My Drive/Colab Notebooks/omniglot/train_nmse_'+fig_id+'.jpg'
            plt.savefig(fig_name)  

        if step % 500 == 0:
            
            accs_maml = []
            accs_metasgd = []
            accs_krmaml = []
            accs_krmaml2 = []
            accs_krmaml_cos = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc_maml = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    test_acc_metasgd = metasgd.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    test_acc_krmaml = krmaml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    #test_acc_krmaml2 = krmaml2.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    test_acc_krmaml_cos = krmaml_cos.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs_maml.append( test_acc_maml )
                    accs_metasgd.append( test_acc_metasgd )
                    accs_krmaml.append( test_acc_krmaml )
                    #accs_krmaml2.append( test_acc_krmaml2 )
                    accs_krmaml_cos.append( test_acc_krmaml_cos )

            # [b, update_step+1]
            accs_maml = torch.mean(torch.as_tensor(accs_maml),0) #np.array(accs_maml).mean(axis=0).astype(np.float16)
            accs_metasgd = torch.mean(torch.as_tensor(accs_metasgd),0)#np.array(accs_metasgd).mean(axis=0).astype(np.float16)
            accs_krmaml = torch.mean(torch.as_tensor(accs_krmaml),0) #np.array(accs_krmaml).mean(axis=0).astype(np.float16)
            #accs_krmaml2 = torch.mean(torch.as_tensor(accs_krmaml2),0) #np.array(accs_krmaml).mean(axis=0).astype(np.float16)
        
            accs_krmaml_cos = torch.mean(torch.as_tensor(accs_krmaml_cos),0) #np.array(accs_krmaml_cos).mean(axis=0).astype(np.float16)
            
            Xaxis2.append(step)
            NMSE_test_maml.append(accs_maml)
            NMSE_test_krmaml.append(accs_krmaml)
            #NMSE_test_krmaml2.append(accs_krmaml2)
            NMSE_test_metasgd.append(accs_metasgd)
            NMSE_test_krmaml_cos.append(accs_krmaml_cos)
            print('Test acc maml:', accs_maml)
            print('Test acc metasgd:', accs_metasgd)
            print('Test acc krmaml:', accs_krmaml)
            #print('Test acc krmaml2:', accs_krmaml2)
            print('Test acc krmaml_cos:', accs_krmaml_cos)
            


            plt.clf()
            plt.plot(Xaxis2,NMSE_test_maml, label='maml')
            plt.plot(Xaxis2,NMSE_test_krmaml, label='krmaml-gaussian-500')
            #plt.plot(Xaxis2,NMSE_test_krmaml2, label='krmaml-gaussian-5000')
            plt.plot(Xaxis2,NMSE_test_krmaml_cos, label='krmaml-cosine')
            plt.plot(Xaxis2,NMSE_test_metasgd, label='metasgd')
            plt.legend()
    
            plt.legend()
            fig_name='/content/drive/My Drive/Colab Notebooks/omniglot/test_nmse_'+fig_id+'.jpg'
            plt.savefig(fig_name)  



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=1)
    argparser.add_argument('--reg', type=int, help='regula', default=1e-4)
    argparser.add_argument('--reg2', type=int, help='regula', default=1e-4)
    argparser.add_argument('--reg_cos', type=int, help='regula', default=1e-4)
    argparser.add_argument('--n_rff', type=int, help='n_rff', default=50)
    argparser.add_argument('--n_rff2', type=int, help='n_rff2', default=5000)
    args = argparser.parse_args()



    main(args)