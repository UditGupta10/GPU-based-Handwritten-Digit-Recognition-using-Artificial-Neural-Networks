
# coding: utf-8

# In[1]:


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
#cuda.initialize_profiler()


# In[2]:


# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Number of blocks to be launched on gpu
nblocks = 95
#bias = 1
# flatten 28*28 images to a 784 vector for each image
num_pixels = 784     #X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels)#.astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels)#.astype('float32')
# Add a coloumn of ones in the 0th pos for bias term
X_train=np.insert(X_train, 784, 255, axis=1)
X_test=np.insert(X_test, 784, 255, axis=1)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)#.astype('float32')
#y_test = np_utils.to_categorical(y_test).astype('float32')
num_classes = 10
# weights
#w_ih = np.random.uniform(-0.25,0.25, (nblocks, X_train.shape[1])).astype('float32')  #95*785
#w_ho = np.random.uniform(-0.25,0.25, (num_classes, nblocks+1)).astype('float32')       #10*96
#w_ih = np.random.normal(size = (nblocks, X_train.shape[1])).astype('float32')  #95*785
#w_ho = np.random.normal(size = (num_classes, nblocks+1)).astype('float32')       #10*96
y_1 = np.ones(nblocks+1)#.astype('float32')
delta_1 = np.ones(nblocks+1)#.astype('float32')
#delta_2 = np.ones(num_classes).astype('float32')


# In[3]:


w1 = np.loadtxt('p0.txt')
w2 = np.loadtxt('p1.txt')
w3 = np.insert(w1, 784, w2, axis=0)

w_ih = np.copy(w3.T)#.astype('float32')


# In[4]:


w11 = np.loadtxt('p2.txt')
w12 = np.loadtxt('p3.txt')
w13 = np.insert(w11, 95, w12, axis=0)

w_ho = np.copy(w13.T)#.astype('float32')


# In[5]:


w_ho.shape


# In[6]:


w_ih_2 = np.copy(w_ih)


# #w_ih = 0.001*np.ones((95, 785)).astype('float32')
# #w_ho = np.ones((95, 785))
# w_ih_2 = np.copy(w_ih)

# #X_train[0] = np.linspace(0,0.005, num=785).astype('float32')

# In[7]:


X_train.dtype


# # Allocate memory on GPU

# In[8]:


X_train_gpu = cuda.mem_alloc(X_train.nbytes)
y_train_gpu = cuda.mem_alloc(y_train.nbytes)
#X_test_gpu = cuda.mem_alloc(X_test.nbytes)
#y_test_gpu = cuda.mem_alloc(y_test.nbytes)
w_ih_gpu = cuda.mem_alloc(w_ih.nbytes)
#w_ih_2_gpu = cuda.mem_alloc(w_ih_2.nbytes)
w_ho_gpu = cuda.mem_alloc(w_ho.nbytes)

#eta_1_gpu = cuda.mem_alloc(eta_1.nbytes)
y_1_gpu = cuda.mem_alloc(y_1.nbytes)
delta_1_gpu = cuda.mem_alloc(delta_1.nbytes)
#eta_2_gpu = cuda.mem_alloc(eta_2.nbytes)
#y_2_gpu = cuda.mem_alloc(y_2.nbytes)
#delta_2_gpu = cuda.mem_alloc(delta_2.nbytes)
#i_gpu = cuda.mem_alloc(8*i.nbytes)
#lr_gpu = cuda.mem_alloc(lr.nbytes)


# In[9]:


cuda.memcpy_htod(X_train_gpu, X_train)
cuda.memcpy_htod(y_train_gpu, y_train)
cuda.memcpy_htod(w_ih_gpu, w_ih)
cuda.memcpy_htod(w_ho_gpu, w_ho)
#cuda.memcpy_htod(i_gpu, i)
#cuda.memcpy_htod(lr_gpu, lr)


# In[10]:


mod = SourceModule("""

    __device__ float sigmoid(float x);
    __device__ float sigmoid(float x){
         return (1.0/(1+expf(-1.0*x)));
    }
  __global__ void ffwd1(int counter, float *X, float *w, float *y_1)
  {
    int j = blockIdx.x;
    int i = threadIdx.x;
    int size_in = blockDim.x;
    __shared__ float temp[785];
    
    //w2[i + j*size_in] = X[i] * w1[i + j*size_in];
    temp[i] = X[counter*size_in + i] * w[i + j*size_in];   //counter*size_in + 
    __syncthreads();
    unsigned int s = 512;
    if(i<273){
        temp[i] += temp[i+s];
    }
    __syncthreads();
    for(s=256; s>0; s>>=1){
        if(i<s){
            temp[i] += temp[i+s];
        }
        __syncthreads();
    }
    if(i==0){
        y_1[j] = tanhf(temp[0]);
        y_1[95] = 1;
    }
  }
  
  __global__ void ffwd2(int counter, float *y_1, float *w, float *d, float *delta_1)
  {
      int i = threadIdx.x;
      // int size_in = blockDim.x;
      int k = i%96;
      int l = i/96;
      int p = i%10;
      int q = i/10;
      float lr = 0.1;
      __shared__ float temp[960];
      __shared__ float w_temp[960];
      __shared__ float y_1_temp[96];
      __shared__ float delta_2_temp[10];
      __shared__ float delta_1_temp[960];
      
      w_temp[i] = w[i];
      __syncthreads();
      if(i<96){
          y_1_temp[i] = y_1[i];
          }
      __syncthreads();
      
      temp[i] = y_1_temp[k]*w_temp[i];     //Each thread does 1 multiplication.
      
       //Reduce
       unsigned int s = 64;
        if(i>=l*96 && i<(l*96 + 32)){
            temp[i] += temp[i+s];
        }
         __syncthreads();
        for(s=32; s>0; s=s/2){
            if(i>=l*96 && i<(l*96 + s)){
                temp[i] += temp[i+s];
            }
            __syncthreads();
        }
        // temp[0] = eta_2[0]; temp[96] = eta_2[1]
        
        if(i<10){
            float y_pred = tanhf(temp[i*96]);
            delta_2_temp[i] = (y_pred-d[counter*10 + i])*(1-y_pred*y_pred);   //counter*10 + 
            //delta_2[i] = delta_2_temp[i];
        }
       __syncthreads();
       
       //w_temp[i] -= lr*delta_2_temp[l]*y_1_temp[k];
       //__syncthreads();
       
       //update delta_1
       delta_1_temp[i] = delta_2_temp[p] * w_temp[q+96*p];
       
       unsigned int a = 8;
        if(i>=q*10 && i<(q*10 + 2)){
            delta_1_temp[i] += delta_1_temp[i+a];
        }
        __syncthreads();
        for(a=4; a>0; a=a/2){
            if(i>=q*10 && i<(q*10 + a)){
                delta_1_temp[i] += delta_1_temp[i+a];
            }
            __syncthreads();
        }
        
        if(i<95){
            delta_1[i] = delta_1_temp[i*10] * (1-y_1_temp[i]*y_1_temp[i]);
        }
       __syncthreads();
       
       //update weights
       w[i] = w_temp[i] - lr*delta_2_temp[l]*y_1_temp[k];    // Each thread updates 1 weight
      }
      
    __global__ void bkwd1(int counter, float *delta_1, float *w, float *X){
        int j = blockIdx.x;
        int i = threadIdx.x;
        int size_in = blockDim.x;
        float lr = 0.1;
        
        w[i+j*size_in] -= lr*delta_1[j]*X[counter*size_in + i];  //counter*size_in + 
    }
  
  """)


# In[11]:


get_ipython().run_cell_magic('time', '', 'for l in range(5):\n    for i in np.arange(60000):#, dtype=\'int32\'):   #60,000 times  X_train.shape[0]\n        #cuda.memcpy_htod(i_gpu, i)\n        c=np.int32(i)\n        func1 = mod.get_function("ffwd1")\n        func1(c, X_train_gpu, w_ih_gpu, y_1_gpu, block=(785, 1, 1), grid=(95,1)) #785 threads, 95 blocks\n        #y_1_after = np.empty_like(y_1).astype(\'float32\')\n        #cuda.memcpy_dtoh(y_1_after, y_1_gpu)\n        #w_ih_2_after = np.empty_like(w_ih_2).astype(\'float32\')\n        #cuda.memcpy_dtoh(w_ih_2_after, w_ih_2_gpu)\n        func2 = mod.get_function("ffwd2")\n        func2(c, y_1_gpu, w_ho_gpu, y_train_gpu, delta_1_gpu, block=(960, 1, 1), grid=(1,1)) #960 threads, 1 block\n        \n        func3 = mod.get_function("bkwd1")\n        func3(c, delta_1_gpu, w_ih_gpu, X_train_gpu, block=(785, 1, 1), grid=(95,1))  #785 threads, 95 blocks')


# In[12]:


w_ih_after = np.empty_like(w_ih)#.astype('float32')
cuda.memcpy_dtoh(w_ih_after, w_ih_gpu)
w_ho_after = np.empty_like(w_ho)#.astype('float32')
cuda.memcpy_dtoh(w_ho_after, w_ho_gpu)


# w_ih_after = np.copy(w_ih)
# w_ho_after = np.copy(w_ho)

# In[ ]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[13]:


dummy = np.tanh(X_train @ w_ih_after.T)
#dummy = np.dot(w_ih_after,X_train[0])


# In[14]:


dummy


# In[ ]:


dummy = np.insert(dummy, 95, 1, axis=1)


# In[ ]:


dummy.shape


# In[ ]:


d_2 = np.tanh(dummy @ w_ho_after.T)

#soft = numpy.empty(d_2.shape)


# In[ ]:


d_2[45]


# In[ ]:


y_train[45]


# In[ ]:


#ypred = numpy.empty(y_test)

y_pred = np.argmax(d_2, axis=1)


# In[ ]:


y_train_2 = np.argmax(y_train, axis=1)
np.sum(y_pred == y_train_2)

