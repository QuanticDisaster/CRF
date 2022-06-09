"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
from crfasrnn.crfrnn import CrfRnn
#from crfasrnn.fcn8s import Fcn8s


class CrfRnnNet(nn.Module):#Fcn8s):
    """
    The full CRF-RNN network with the FCN-8s backbone as described in the paper:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
    """

    def __init__(self, num_labels=0):
        super(CrfRnnNet, self).__init__()
        self.crfrnn = CrfRnn(num_labels=num_labels, num_iterations=10)
        
        self.testing = nn.Linear(num_labels,num_labels)

    def forward(self, data):
        #out are unary potential
        
        #points are positions + features
        points = torch.cat([data.pos, data.x], axis=1)
        out = data.unary_potentials
                
        # Plug the CRF-RNN module at the end
        updated_logits = self.testing(out)#self.crfrnn(points, out)
        
        return updated_logits