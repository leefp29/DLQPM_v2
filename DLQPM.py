import os
import torch
import torch.nn as nn
from torch._C import device
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch.optim as optim
import time
from torch.optim import lr_scheduler
plt.style.use(["science", "notebook", "no-latex"])



seed_ini = np.random.randint(0, 2 ** 31 -1)
print(seed_ini)
# seed_ini = 924953066
def setup_seed(seed):
#     random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
setup_seed(seed_ini)

# visualize the training process
def plot_Delta(epoch, Tarr, DeltaPred, TraceAnomaly, Tc=0.155):
    if epoch % 500 != 0: return
    DeltaPred = DeltaPred.cpu().detach().numpy()
    TraceAnomaly = TraceAnomaly.cpu().detach().numpy()
    plt.plot(Tarr/Tc, DeltaPred.flatten()/Tarr**4, 'r-', label="network")
    plt.plot(Tarr/Tc, TraceAnomaly.flatten()/Tarr**4, 'b--', label="input")
    plt.legend(loc='best')
    plt.xlabel(r"$T/T_c$",loc="center")
    plt.ylabel(r"$(\varepsilon - 3 P)/T^4$",loc="center")
    plt.savefig("pic_mu/Delta_vs_T_epoch%s.jpg"%epoch)
    plt.close()
    
def plot_ed(epoch, Tarr, EdPred, Earr, Tc=0.155):
    """
    inupt: epoch
    return: T as a function of enrgy density
    """
    if epoch % 500 != 0: return
    EdPred = EdPred.cpu().detach().numpy()
    plt.plot(Tarr/Tc, EdPred.flatten(), 'r-', label='network')
    plt.plot(Tarr/Tc, Earr.cpu().detach().numpy(), 'b--', label='input')
    plt.xlabel(r"$T\ {\rm [GeV]}$",loc="center")
    plt.ylabel(r"$\varepsilon\ {\rm [GeV]^{4}}$",loc="center")
    plt.legend(loc='best')
    plt.savefig("pic_mu/Ed_vs_T_epoch%s.jpg"%epoch)
    plt.close()
    
def plot_pr(epoch, Tarr, PrPred, Parr, Tc=0.155):
    """
    inupt: epoch
    return: T as a function of Pressure
    """
    if epoch % 500 != 0: return
    PrPred = PrPred.cpu().detach().numpy()
    plt.plot(Tarr/Tc, PrPred.flatten(), 'r-', label='network')
    plt.plot(Tarr/Tc, Parr.cpu().detach().numpy(), 'b--', label='input')
    plt.xlabel(r"$T\ {\rm [GeV]}$",loc="center")
    plt.ylabel(r"$P\ {\rm [GeV]^{4}}$",loc="center")
    plt.legend(loc='best')
    plt.savefig("pic_mu/Pr_vs_T_epoch%s.jpg"%epoch)
    plt.close()

def plot_entropy(epoch, Tarr, SPred, Sarr, Tc=0.155):
    """
    inupt: epoch
    return: T as a function of entropy
    """
    if epoch % 500 != 0: return
    SPred = SPred.cpu().detach().numpy()
    plt.plot(Tarr/Tc, SPred.flatten(), 'r-', label='network')
    plt.plot(Tarr/Tc, Sarr.cpu().detach().numpy(), 'b--', label='input')
    plt.xlabel(r"$T\ {\rm [GeV]}$",loc="center")
    plt.ylabel(r"$S\ {\rm [GeV]^{3}}$",loc="center")
    plt.legend(loc='best')
    plt.savefig("pic_mu/S_vs_T_epoch%s.jpg"%epoch)
    plt.close()
    
def plot_Chib2(epoch, Tarr, CB2Pred, CB2):
    """
    inupt: epoch
    return: T as a function of entropy
    """
    if epoch % 500 != 0: return
    CB2Pred = CB2Pred.cpu().detach().numpy()
    plt.plot(Tarr, CB2Pred.flatten(), 'r-', label='network')
    plt.scatter(Tarr, CB2.cpu().detach().numpy(), c = 'k', label='input')
    plt.xlabel(r"$T\ {\rm [GeV]}$",loc="center")
    plt.ylabel(r"$\chi_B^2$",loc="center")
    plt.legend(loc='best')
    plt.savefig("pic_mu/CB2_vs_T_epoch%s.jpg"%epoch)
    plt.close()
def plot_Chib4(epoch, Tarr, CB4Pred, CB4):
    """
    inupt: epoch
    return: T as a function of entropy
    """
    if epoch % 500 != 0: return
    CB4Pred = CB4Pred.cpu().detach().numpy()
    plt.plot(Tarr, CB4Pred.flatten(), 'r-', label='network')
    plt.scatter(Tarr, CB4.cpu().detach().numpy(), c = 'k', label='input')
    plt.xlabel(r"$T\ {\rm [GeV]}$",loc="center")
    plt.ylabel(r"$\chi_B^4$",loc="center")
    plt.legend(loc='best')
    plt.savefig("pic_mu/CB4_vs_T_epoch%s.jpg"%epoch)
    plt.close()
    
def plot_Chib6(epoch, Tarr, CB6Pred, CB6):
    """
    inupt: epoch
    return: T as a function of entropy
    """
    if epoch % 500 != 0: return
    CB6Pred = CB6Pred.cpu().detach().numpy()
    plt.plot(Tarr, CB6Pred.flatten(), 'r-', label='network')
    plt.scatter(Tarr, CB6.cpu().detach().numpy(), c = 'k', label='input')
    plt.xlabel(r"$T\ {\rm [GeV]}$",loc="center")
    plt.ylabel(r"$\chi_B^6$",loc="center")
    plt.legend(loc='best')
    plt.savefig("pic_mu/CB6_vs_T_epoch%s.jpg"%epoch)
    plt.close()
    
def plot_sc(epoch,SPred):
    """
    inupt: epoch
    return: T as a function of entropy
    """
    if epoch % 500 != 0: return
    SPred = SPred.cpu().detach().numpy()
    plt.imshow(SPred.reshape(15,15))
    plt.colorbar()
    plt.savefig("pic_mu/Sc_vs_T_epoch%s.jpg"%epoch)
    plt.close()


# In[5]:


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        
        self.l1 = nn.Linear(channels,channels)
        self.l2 = nn.Linear(channels,channels)
    
    def forward(self, x):
        y = torch.sigmoid(self.l1(x))
        y = self.l2(y)
        return x+y
    
class Net_Mass(nn.Module):
    def __init__(self, input,  output):
        """
        input: tensor and shape (None, 1)
        NL: the number of layers
        NN: the number of neurons
        activate function: sigmoid
        output: tensor and shape (None, 1)
        """
        super(Net_Mass,self).__init__()
        self.input_layer1 = nn.Linear(input, 16)
        self.hidden_layers0= nn.Linear(16, 32)
        self.hidden_layers1 = ResidualBlock(32)
        self.hidden_layers2 = ResidualBlock(32)
        self.hidden_layers3 = ResidualBlock(32)
        self.hidden_layers4 = ResidualBlock(32)
        self.hidden_layers5 = ResidualBlock(32)
        self.hidden_layers6 = ResidualBlock(32)
        self.hidden_layers7 = ResidualBlock(32)
        self.hidden_layers8 = nn.Linear(32,16)
        self.output_layer = nn.Linear(16,output)
        
    def forward(self,x):
        
        o = self.act(self.input_layer1(x))
        o = self.act(self.hidden_layers0(o))
        o = self.act(self.hidden_layers1(o))
        o = self.act(self.hidden_layers2(o))
        o = self.act(self.hidden_layers3(o))
        o = self.act(self.hidden_layers4(o))
        o = self.act(self.hidden_layers5(o))
        o = self.act(self.hidden_layers6(o))
        o = self.act(self.hidden_layers7(o))
        o = self.act(self.hidden_layers8(o))

            
        opt = self.output_layer(o)
        opt = self.act1(opt)
        return opt
    
    def act(self,x):
        return x * torch.sigmoid(x)
    
    def act1(self,x):
        return torch.sigmoid(x)
    

# partition function of quark and gluon
def lnZ_q(T,  m,  mu, eta):
    deg = 50
    xk, wk = np.polynomial.laguerre.laggauss(deg=deg)
    pnodes = torch.from_numpy(xk).to(device)
    wnodes = torch.from_numpy(wk).to(device)
#     pnodes = 
    rnodes =  torch.exp(-pnodes)
    psqure = pnodes**2
    E = (psqure + m ** 2) ** 0.5
    E_mu = E - mu
    efactor = torch.exp(-E_mu/T)
    f = efactor * eta
    f = f + 1
    f = torch.log(f)
    f = psqure * eta * f
    f = wnodes * f
    f = f/rnodes
    f = torch.sum(f,1)
#     print(torch.sum(f,1).shape)
    return f.reshape(-1,1)

def lnZ_g(T, m, eta):
    deg = 50
    xk, wk = np.polynomial.laguerre.laggauss(deg=deg)
    pnodes = torch.from_numpy(xk).to(device)
    wnodes = torch.from_numpy(wk).to(device)
    
    rnodes =  torch.exp(-pnodes)
    psqure = pnodes**2
    E = (psqure + m ** 2) ** 0.5
    efactor = torch.exp(-E/T)
    f = efactor * eta
    f = f + 1
    f = torch.log(f)
    f = psqure*eta * f
    f = wnodes * f
    f = f/rnodes
    f = torch.sum(f,1)
    return  f.reshape(-1,1)

# training the mass of quark and gluon

def Mass_Tmu_train(learning_rate, epochs, path, device):
    """
    return: loss, and save model
    """

    Net1 = Net_Mass(2,1).to(device)
    Net2 = Net_Mass(2,1).to(device)
    Net3 = Net_Mass(2,1).to(device)
    
    opt1 = optim.AdamW(Net1.parameters(),lr = learning_rate,betas=(0.9,0.999))
    opt2 = optim.AdamW(Net2.parameters(),lr = learning_rate,betas=(0.9,0.999))
    opt3 = optim.AdamW(Net3.parameters(),lr = learning_rate,betas=(0.9,0.999))
    
    lr_step_size =2000

    schedular1 = lr_scheduler.StepLR(opt1, step_size=lr_step_size, gamma=0.9)    
    schedular2 = lr_scheduler.StepLR(opt2, step_size=lr_step_size, gamma=0.9)    
    schedular3 = lr_scheduler.StepLR(opt3, step_size=lr_step_size, gamma=0.9)    
    
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            
    Net1.apply(init_normal)
    Net1.train()
    Net2.apply(init_normal)
    Net2.train()    
    Net3.apply(init_normal)
    Net3.train()    
    

    
    ########## loss function
    def loss_a(T, muB):

        T.requires_grad = True
        muB.requires_grad = True
        
        
        mu_ud = 1/3 * muB
        mu_s = 1/3 * muB
        
        inp1 = torch.cat((T, muB),1)
        inp2 = torch.cat((T, muB),1)
        inp3 = torch.cat((T, muB),1)

        
        # Obtain Mass
        Mud = Net1(inp1)
        Ms = Net2(inp2)
        Mgluon = Net3(inp3)
    
        
        coef =  1 / (2 * torch.pi**2)   
        ud_dof = 2 * 2 * 3  #u,d * q, qbar * spin_dof * color_dof
        s_dof = 2 * 3
        gluon_dof = 8 * 2       # color dof * polarization dof       
        
        lnZ_ud =  ud_dof * coef * lnZ_q(T, Mud, mu_ud, 1) + ud_dof * coef * lnZ_q(T, Mud, -mu_ud, 1)

        
        lnZ_s = s_dof * coef * lnZ_q(T, Ms, mu_s, 1) + s_dof * coef * lnZ_q(T, Ms, -mu_s, 1)
 

        lnZ_gluon = gluon_dof * coef * lnZ_g(T, Mgluon, -1)

 
        lnZ_tot = lnZ_ud + lnZ_s + lnZ_gluon
   

        dlnZdT =  torch.autograd.grad(lnZ_tot, T, grad_outputs=torch.ones_like(T).to(device),create_graph=True)[0]
        dlnZdmu = torch.autograd.grad(lnZ_tot, muB, grad_outputs=torch.ones_like(muB).to(device),create_graph=True)[0]
        d2lnZdmu = torch.autograd.grad(dlnZdmu, muB, grad_outputs=torch.ones_like(muB).to(device),create_graph=True)[0]
        d3lnZdmu = torch.autograd.grad(d2lnZdmu, muB, grad_outputs=torch.ones_like(muB).to(device),create_graph=True)[0]
        d4lnZdmu = torch.autograd.grad(d3lnZdmu, muB, grad_outputs=torch.ones_like(muB).to(device),create_graph=True)[0]


        Chi_B2 = d2lnZdmu/ T
        Chi_B4 = d4lnZdmu * T

        
        
        nd = T * dlnZdmu # particle number density
        pr = T * lnZ_tot  # pressure
        s_pred = T * dlnZdT  +  lnZ_tot # entropy
        ed = T * s_pred - pr + muB * nd  #  energy density  


        
        Delta = (ed - 3 * pr)
        
        mass_loss_1 = torch.where(T >(2.5 * 0.150), torch.abs(Mgluon/Mud - 1.5), torch.tensor([0.]).to(device)) # HTL mass constraint
        mass_loss_2 = torch.where(T >(2.5 * 0.150), torch.abs((Ms  - Mud)/0.09 - 1), torch.tensor([0.]).to(device))
        L_mass =  (mass_loss_1 + mass_loss_2)

        sam = len(s_true)
        loss_mae = nn.L1Loss()
        loss1 = loss_mae(s_pred[0:sam], s_true).to(device)
        loss2 = loss_mae(Delta[0:sam]/T[0:sam], D_true/T[0:sam]).to(device)
        loss3 = loss_mae(Chi_B2[sam:], Chi_b2_true).to(device)
        loss4 = loss_mae(Chi_B4[sam:], Chi_b4_true).to(device)
        loss6 = loss_mae(L_mass , torch.zeros_like(L_mass)).to(device)

        
        loss_tot = loss1+ loss2 + loss3  + loss4 + loss6 
  
        return loss_tot, loss1, loss2,  loss3, loss4, loss6, ed, pr, s_pred, Delta, Chi_B2, Chi_B4

    tic = time.time()
    Loss_list = []
    Loss_mean = 1e5

    df1 = pd.read_csv("./data/hotqcd_1407.6387_noerrbar_allT.csv")
    df2 = pd.read_csv("./data/fig1_data.csv")
 
    s_true = df1["s/T^3"] * df1["T"] ** 3
    E_true = df1["E/T^4"] * df1["T"] ** 4
    P_true = df1["P/T^4"] * df1["T"] ** 4
    D_true = E_true - 3 * P_true
    Tem = np.concatenate((df1["T"], df2["TMeV"]/1000))

    Chi_b2_true = df2["BQS200"] 
    Chi_b4_true = df2["BQS400"] 
    sam = len(s_true)

    Tem = torch.FloatTensor(Tem).reshape(-1,1).to(device)     
    D_true = torch.FloatTensor(D_true).reshape(-1,1).to(device)     
    s_true = torch.FloatTensor(s_true).reshape(-1,1).to(device)     
    P_true = torch.FloatTensor(P_true).reshape(-1,1).to(device)     
    E_true = torch.FloatTensor(E_true).reshape(-1,1).to(device)     
    Chi_b2_true = torch.FloatTensor(Chi_b2_true).reshape(-1,1).to(device)
    Chi_b4_true = torch.FloatTensor(Chi_b4_true).reshape(-1,1).to(device)
    muB = torch.zeros_like(Tem).to(device)
    

    

    # Each step of the training process:
    for epoch in range(epochs):
        Loss = []


        Net1.zero_grad()        
        Net2.zero_grad()        
        Net3.zero_grad()        
        loss, loss1, loss2, loss3 ,loss4,  loss6, energy, pre, entropy, Trace, cb2, cb4 = loss_a(Tem, muB)

        plot_entropy(epoch,df1["T"],entropy[0:sam], s_true)
        plot_Delta(epoch,df1["T"],Trace[0:sam],D_true)
        plot_ed(epoch,df1["T"],energy[0:sam],E_true)
        plot_pr(epoch,df1["T"],pre[0:sam],P_true)
        plot_Chib2(epoch,df2["TMeV"]/1000,cb2[sam:], Chi_b2_true)
        plot_Chib4(epoch,df2["TMeV"]/1000,cb4[sam:],Chi_b4_true)

        # Back propagation and optimizer
        loss.backward()
        opt1.step()
        opt2.step()
        opt3.step()
        schedular1.step()
        schedular2.step()
        schedular3.step()
        lr = schedular1.get_last_lr()[0]

        # print the loss value
        Loss.append(loss.item())
        print('Train Epoch:{} learning rate:{:.4e}, Loss:{:.6e}, Loss1:{:.6e}, Loss2:{:.6e}'.format(epoch+1,lr,loss.item(),loss1.item(),loss2.item()))
        Loss = np.array(Loss)
        
        # check the loss and save the model        
        if np.mean(Loss) < Loss_mean:
            checkpoint1 = {
                    'epoch': epoch + 1,
                    'state_dict': Net1.state_dict(),
                    'optimizer': opt1.state_dict(),
                    'loss': loss.item()
                    }
            checkpoint2 = {
                    'epoch': epoch + 1,
                    'state_dict': Net2.state_dict(),
                    'optimizer': opt2.state_dict(),
                    'loss': loss.item()
                    }     
            checkpoint3 = {
                    'epoch': epoch + 1,
                    'state_dict': Net3.state_dict(),
                    'optimizer': opt3.state_dict(),
                    'loss': loss.item()
                    }        
            torch.save(checkpoint1,path+"Mud_model.pt")
            torch.save(checkpoint2,path+"Ms_model.pt")
            torch.save(checkpoint3,path+"Mg_model.pt")
            print("save model")
            Loss_mean = np.mean(Loss)
            
        Loss_list.append(np.mean(Loss))
    
    toc = time.time()
    trainTime = toc - tic
    print("Training time = ", trainTime)
    np.save(path + 'Loss_all',np.array(Loss_list))


# build the folder
def mkdir(path):

    folder = os.path.exists(path)

    if not folder:                   
        os.makedirs(path)            
        print ("---  new folder...  ---")
        print ("---  OK  ---")

    else:
        print ("---  There is this folder!  ---")
mkdir('data')
mkdir('model')
mkdir('pic')


# parameters    
epochs = 50000
path_data = "./data"
path_model = "./model/"
path_pic = "./pic/"
learning_rate = 1e-3
device = torch.device('cpu') # cpu or cuda

# run the training
Mass_Tmu_train(learning_rate, epochs, path_model, device)
