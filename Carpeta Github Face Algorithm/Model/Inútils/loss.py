from torch import linalg
import torch.nn.functional as F
import torch



def SST_loss(A, P, N):
    A_P_distance=-torch.linalg.vector_norm(A-P,dim=1)**2
    A_N_distance=-torch.linalg.vector_norm(A-N,dim=1)**2
    N_P_distance=-torch.linalg.vector_norm(N-P,dim=1)**2
    denominator=torch.exp(A_P_distance)+torch.exp(1/2*torch.add(A_N_distance,N_P_distance))
    print(torch.exp(1/2*torch.add(A_N_distance,N_P_distance)))
    loss=torch.exp(A_P_distance)/denominator
    return loss.mean()


A=torch.rand((16,1280))
B=torch.rand((16,1280))
C=torch.rand((16,1280))


SST_loss(A,B,C)

def SST_loss(A, P, N):
    A_P_distance=-torch.linalg.vector_norm(())
    print((A_P_distance))
    A_N_distance=-torch.sum(torch.exp2(torch.subtract(A,N)),dim=1)
    N_P_distance=-torch.sum(torch.exp2(torch.subtract(N,P)),dim=1)
    denominator=torch.exp(A_P_distance)+torch.exp(1/2*torch.add(A_N_distance,N_P_distance))
    loss=torch.exp(A_P_distance)/denominator
    return loss.mean()




