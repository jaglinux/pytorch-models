# to train the model
## install torch
python cifar10/cifar10.py

# training time on single GPU
device found  cuda:0
PyTorch is using ROCm
Files already downloaded and verified
Files already downloaded and verified
Model is on CUDA  cuda:0
Accuracy on the test set: 72 %

real    1m0.721s
user    1m17.927s
sys     0m7.536s

# training time on CPU (its more than GPU thought the accuracy is same as expected)
device found  cpu
PyTorch is using ROCm
Files already downloaded and verified
Files already downloaded and verified
Model is on CPU
Accuracy on the test set: 71 %

real    4m47.465s
user    313m58.172s
sys     8m34.253s
