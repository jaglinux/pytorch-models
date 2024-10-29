# to train the model
## install torch
python cifar10/cifar10.py

# training time on single GPU
device found  cuda:0n <br>
PyTorch is using ROCm <br>
Files already downloaded and verified <br>
Files already downloaded and verified <br>
Model is on CUDA  cuda:0 <br>
Accuracy on the test set: 72 %  <br>

real    1m0.721s <br>
user    1m17.927s <br>
sys     0m7.536s <br>

# training time on CPU (its more than GPU thought the accuracy is same as expected)
device found  cpu <br>
PyTorch is using ROCm <br>
Files already downloaded and verified <br>
Files already downloaded and verified <br>
Model is on CPU <br>
Accuracy on the test set: 71 % <br>

real    4m47.465s <br>
user    313m58.172s <br>
sys     8m34.253s <br>
