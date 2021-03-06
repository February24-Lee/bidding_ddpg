from torch.utils import data

class logData(data.Dataset):
    def __init__(self, array, c,  m1, m2):
        self.array  = array
        self.c      = c
        self.m1     = m1
        self.m2     = m2

    def __len__(self):
        assert self.array.shape[0] == self.c.shape[0]
        return self.array.shape[0]

    def __getitem__(self, i):
        return self.array[i, :], self.c[i, :], self.m1[i, :], self.m2[i, :]


class logData_test(data.Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, i):
        return self.array[i, :]