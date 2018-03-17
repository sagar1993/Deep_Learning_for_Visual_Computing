import numpy as np

def conv_forward(X, W, b, stride=1, padding=1):
    
    n, c, h, w = X.shape
    f, c, hh, ww = W.shape 
    padded_x = (np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant'))
    
    h_out = (h + 2 * padding - hh) // stride + 1 
    w_out = (w + 2 * padding - ww) // stride + 1
    
    out = np.zeros((n, f, h_out, w_out))
    for _n in range(n):
        for _f in range(f):
            for _h_out in range(h_out):
                for _w_out in range(w_out):
                    out[_n, _f, _h_out, _w_out] = np.sum(W * padded_x[_n, : , _h_out*stride: _h_out*stride +hh, _w_out*stride : _w_out*stride + ww]) + b[_f]
    cache = (X, W, b, stride, padding)
    return out, cache
    

def conv_backward(dout, cache):
    
    X, W, b, stride, padding = cache
    padded_x = (np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant'))
    
    n, c, h, w = X.shape
    f, c, hh, ww = W.shape 
    n, f, h_out, w_out = dout.shape
    
    dx_t = np.zeros_like(padded_x)
    dw = np.zeros_like(W)
    db = np.zeros_like(b)
    
    for _f in range(f):
        db[_f] += np.sum(dout[:,_f,:,:])
    
    for _n in range(n):
        for _f in range(f):
            for _h_out in range(h_out):
                for _w_out in range(w_out):
                    dw[_f] = dout[_n, _f, _h_out, _w_out] * padded_x[_n, : , _h_out*stride: _h_out*stride +hh, _w_out*stride : _w_out*stride + ww]
                    
    
    for _n in range(n):
        for _f in range(f):
            for _h_out in range(h_out):
                for _w_out in range(w_out):
                    dx_t[_n, : , _h_out*stride: _h_out*stride +hh, _w_out*stride : _w_out*stride + ww] = dout[_n, _f, _h_out, _w_out] * dw[_f]

    
    dx = dx_t[:,:,padding:h+padding]    
    return dx, dw, db


def max_pool_forward(x, height, width, stride):
    out = None
    
    n,c,h,w = x.shape
    
    h_out = np.int(((h - height) // stride) + 1)
    w_out = np.int(((w - width) // stride) + 1)
    
    out = np.zeros([n,c,h_out, w_out])
    
    for _n in range(n):
        for _f in range(c):
            for _h_out in range(h_out):
                for _w_out in range(w_out):
                        out[_n, _f, _h_out, _w_out] = np.max(x[_n, _f, _h_out*stride:_h_out*stride+height, _w_out*stride:_w_out*stride+width])

    cache = x, height, width, stride
    return out, cache
                        
def max_pool_backward(dout, cache):
    x, height, width, stride = cache
    
    n,c,h,w = x.shape
    _,_,h_out, w_out = dout.shape
    dx = np.zeros_like(x)
    
    for _n in range(n):
        for _f in range(c):
            for _h_out in range(h_out):
                for _w_out in range(w_out):
                    index = np.argmax(x[_n, _f, _h_out*stride:_h_out*stride+height, _w_out*stride:_w_out*stride+width])
                    coord = np.unravel_index(index, [height,width])
                    dx[_n, _f, _h_out*stride:_h_out*stride+height, _w_out*stride:_w_out*stride+width][coord] = dout[_n, _f, _h_out, _w_out]

    return dx
                    
def dropout_forward(x, p, mode, seed=0):
    np.random.seed(seed)
    if mode == "train":
        mask = np.random.random_sample(x.shape) >= p
        scale = 1/(1-p)
        mask = mask * scale
        out = x*mask
    else:
        out = x
    cache = p, mode, mask
    return out, cache
    

def dropout_backward(dout, cache):
    p, mode, mask = cache
    dx = None
    if mode == "train":
        dx = dout*mask
    else:
        dx = dout
    return dx


def nn_forward(x,w,b):
    out = None
    n = x.shape[0]
    x_reshaped = x.reshape(n,-1)
    z = x_reshaped.dot(w) + b
    cache = x,w,b
    return z, cache

def nn_backward(dout, cache):
    x, w, b = cache
    n = x.shape[0]
    x_reshaped = x.reshape(n,-1)
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)
    dw = np.dot(x_reshaped.T,dout)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db

def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def relu_forward(x):
    out = None
    out = x.copy()
    out[out < 0] = 0
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    relu_mask = (x >= 0)
    dx = dout * relu_mask
    return dx