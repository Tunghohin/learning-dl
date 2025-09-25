import torch
import utils.Timer as Timer

n = 1000000
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def timer_example():
    a_cpu = torch.randint(0, n, (n,), device='cpu')
    b_cpu = torch.randint(0, n, (n,), device='cpu')
    c = torch.zeros(n, device='cpu')
    a_gpu = torch.randint(0, n, (n,), device=device)
    b_gpu = torch.randint(0, n, (n,), device=device)
    d = torch.zeros(n, device=device)
    
    timer = Timer()
    for i in range(n):
        c[i] = torch.exp(a_cpu[i]) + torch.log(b_cpu[i])
    print(f'Loop time: {timer.stop():.6f}s')

    timer.reset()
    d = torch.exp(a_gpu) + torch.log(b_gpu)
    print(f'Vectorized time: {timer.stop():.6f}s')
