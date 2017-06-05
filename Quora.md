# deep learning学习中遇到的问题

1. [How view() method works for tensor in torch
](https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch)

2. [为什么要用Variable,如final = model(Variable(image))](https://github.com/jcjohnson/pytorch-examples)
    * We wrap our PyTorch Tensors in Variable objects; a Variable represents a node in a computational graph. If x is a Variable then x.data is a Tensor, and x.grad is another Variable holding the gradient of x with respect to some scalar value.
    * using Variables defines a computational graph, allowing you to automatically compute gradients.
    
3. [numpy squeeze函数的作用](http://blog.csdn.net/pipisorry/article/details/39496831)
    * 将所有维度为1的维度去掉。这个操作应该等价于a.reshape(-1)
    
4. [[]和（）有什么区别？](http://www.cnblogs.com/deepleo/p/python-list-tuple-dict.html)
    * []列表，其中的元素可以改变，()元组，其中的元素不能改变，其余部分两者很相似。
    