import torch 
a=torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32,device='cuda')
b=torch.sum(a)
print(b)
