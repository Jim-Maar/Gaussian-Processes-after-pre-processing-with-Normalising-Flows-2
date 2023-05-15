import torch
def index_generation(crt_i, max_n, N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l
#[4, 3, 0, 1, 2]
#[5, 4,1,2,3]
#[6,5,2,3,4]
#[7,6,3,4,5]
#[8,7,4,5,6]

input_all=torch.randn([6,120,3,32,32])
max_n=120
N=5
n=3
batches=[]
for i in range(0,120,n):
    #input_batch=input_all[32,i:i+9,:,:]
    batch_now= []
    for j in range(i,i+n):
        #Generate this index for every 5 frames
        ind=index_generation(j,max_n,N, padding='new_info')
        batch_now.append(input_all[:,ind,:,:].unsqueeze(1))
        print(ind)
    batch_now=torch.cat(batch_now,dim=1)
    print(batch_now.shape)