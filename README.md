# CNN with Kolmogorov-Arnold Networks 

In our investigation of CNN architectures, we integrated Kolmogorov-Arnold Networks (KANs) to compare against traditional fully connected (FC) layers. We found that KANs, with their non-linear spline-based transformations, can capture complex patterns more efficiently than FC layers, potentially reducing the need for deeper or more complex network structures. Despite KANs typically having a larger parameter count due to their intricate spline functions, they offer a significant advantage in modeling capabilities. This study highlighted the potential of KANs to outperform standard FC layers in tasks requiring high levels of data interpretation and complexity. Our findings suggest a promising avenue for future research in neural network design, focusing on optimizing KAN configurations to balance parameter efficiency with computational performance. 

## Network Architecture and Parameter Count 
```
CNNKAN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (kan1): KANLinear(
    (base_activation): SiLU()
  )
  (kan2): KANLinear(
    (base_activation): SiLU()
  )
)
conv1.weight: 864
conv1.bias: 32
conv2.weight: 18432
conv2.bias: 64
kan1.base_weight: 1048576
kan1.spline_weight: 8388608
kan1.spline_scaler: 1048576
kan2.base_weight: 2560
kan2.spline_weight: 20480
kan2.spline_scaler: 2560
Total trainable parameters: 10530752
``` 
