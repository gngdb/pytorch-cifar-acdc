import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalLoss(nn.Module):
    def forward(self, x, y):
        # whatever x comes in as, we flatten it
        x = x.view(x.size(0),-1)
        n,d = x.size()
        assert n%2 == 0, "batch size must be divisible by 2, but was %i"%n
        if self.train:
            # split into support and target
            m = n//2
            x_support, x_target = x[:m,:], x[m:,:]
            y_support, y_target = y[:m], y[m:]
            # calculate prototypes
            #if y_support.max() != y_target.max():
            #    import pdb
            #    pdb.set_trace()
            num_classes = max(y_support.max(), y_target.max())+1
            o_support = onehot(y_support, num_classes)
            x_support = x_support.view(m,1,d)
            o_support = o_support.view(m,num_classes,1).float().to(x_support.device)
            sep_by_class = x_support*o_support
            class_counts = o_support.sum(0, keepdim=True)
            class_counts = class_counts + (class_counts < 1e-2).float() # for safety
            assert class_counts.min() > 0
            prototypes = sep_by_class.sum(0, keepdim=True)/class_counts
            # store support prototypes
            self.prototypes = prototypes
        else:
            # then we used the stored prototypes
            prototypes = self.prototypes
            # and everything is target set
            x_target, y_target = x, y
        # use these prototypes to classify target set
        x_target = x_target.view(m,1,d)
        distance = -square(x_target-prototypes).mean(2)
        #distance = F.log_softmax(distance)
        # cross entropy loss
        loss = F.cross_entropy(distance, y_target)
        self.acc = accuracy(distance, y_target)
        # DEBUGGING
        nans = loss[loss != loss]
        #import pdb
        #pdb.set_trace()
        # accuracy 
        return loss


def accuracy(x, y):
    _, predicted = x.max(1)   
    total = y.size(0)   
    correct = predicted.eq(y).sum().item()
    return 100.*correct/total

def onehot(y, num_classes):
    # the shortest answer
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/25
    labels = y
    y = torch.eye(num_classes)
    return y[labels]

def square(x):
    #return torch.pow(x,2)
    sgn = torch.sign(x)
    return sgn*torch.exp(2.*torch.log(torch.abs(x+1e-5)))

if __name__ == '__main__':
    x = torch.randn(100,10)
    y = torch.LongTensor(100).random_(0,10)
    
    criterion = PrototypicalLoss()
    print(criterion(x,y))

    criterion.eval()
    print(criterion(x,y))
