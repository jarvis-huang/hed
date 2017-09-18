import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
import Image
import scipy.io
import os
caffe_root = '/home/jhuang/repo/hed/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

data_root = '/home/jhuang/data/HED-BSDS/'
with open(data_root+'test.lst') as f:
    test_lst = f.readlines()
test_lst = [data_root+x.strip() for x in test_lst]

#Test images must have three color channels.
test_lst = ['/home/jhuang/repo/Edgelets/trunk/img/clean/frm9000_color_histeq.png']
#test_lst = ['/home/jhuang/repo/Edgelets/trunk/img/bright/frm22000_color_histeq.png']
#test_lst = ['/home/jhuang/repo/Edgelets/trunk/img/dark/frm15500_color_histeq.png']
#test_lst = ['/home/jhuang/data/HED-BSDS/test/8068.jpg']
im_lst = []
for i in range(0, len(test_lst)):
    im = Image.open(test_lst[i])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    im_lst.append(in_)

#Visualization
def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(2,(len(scale_lst)+1)/2,i+1)
        plt.imshow(1-scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()    
    
    
idx = 0
in_ = im_lst[idx]
in_ = in_.transpose((2,0,1))

#Testing with cpu or gpu
compute_mode='cpu'
if compute_mode=='gpu':
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

# load net
model_root = '/home/jhuang/repo/hed/examples/hed/'
net = caffe.Net(model_root+'deploy.prototxt', model_root+'hed_pretrained_bsds.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]

# type(fuse) = <type 'numpy.ndarray'>
# fuse.shape = (1198, 1888)
scale_lst = [fuse, out1, out2, out3, out4, out5]
plot_single_scale(scale_lst, 22)
plt.show()
print 'image file =', test_lst[idx]
