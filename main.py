import torch

if __name__ == '__main__':
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(t1 @ t2)
    print((t2 @ t1).shape)