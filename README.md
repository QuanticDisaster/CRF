# CRF

This project is an attempt at integrating a CRF layer as a trainable pytorch layer for points cloud. This implementation is partly adapted from CRF-RNN source code : https://github.com/sadeepj/crfasrnn_pytorch


### Step 1: install requirements

torch, torch-geometric, pandas, matplotlib

### Step 2: Build CRF-RNN custom op

Run `setup.py` inside the `CRF/code/crfasrnn` directory:
```
$ cd CRF/code/crfasrnn
$ python setup.py install 
``` 

Note that the `python` command in the console should refer to the Python interpreter associated with your PyTorch installation. 

### Step 3: download data

Available in the zip file

### Step 4: Run a training
```
$ sh commands.sh
```



