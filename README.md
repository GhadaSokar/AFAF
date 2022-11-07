# Avoiding Forgetting and Allowing Forward Transfer in Continual Learning via Sparse Networks
[ECMLPKDD 2022] [Avoiding Forgetting and Allowing Forward Transfer in Continual Learning via Sparse Networks](https://arxiv.org/pdf/2110.05329.pdf) by by Ghada Sokar, Decebal Constantin Mocanu, and Mykola Pechenizkiy.

# Requirements
* Python 3.6
* Pytorch 1.2
* torchvision 0.4

# Usage
You can use main_CNN.py. 

```
python main.py
```

Options 
```
* --benchmark: oprions (CIFAR10, CIFAR100, mix)
* --class_order: the order of the classes (i.e., '0,1,2,3,4,5,6,7,8,9' for split-CIFAR10 and '1,3,7,9,5,4,0,2,6,8' for sim-CIFAR10)
* --num_tasks: number of tasks in the sequence
* --num_classes_per_task: number of classes in each task
* --knowledge_reuse True: to use candidate neurons in allocation
* --l_reuse: value for l_reuse (reusing full layers)
* --reuse_from: start reusing full layers from task x. x=2 for split-CIFAR10 and sim-CIFAR10 and x=4 for Mix and sim-CIFAR100
* --alloc_prec_conv: percentage of allocated neurons in convolution layers. You can find similar arguments for FC and output.
* --subfree_prec_conv: percentage of free neurons from allocated ones. You can find similar arguments for FC and output.
* --reuse_prec_conv: percentage of candidate neurons from allocated ones. You can find similar arguments for FC and output.
* --freezed_prec_conv: percentage of fixed neurons from allocated ones. You can find similar arguments for FC and output.
* --density_level_conv: density level of connections between allocated neurons. You can find similar arguments for FC and output.
```

# Reference
If you use this code, please cite our paper:
```
@inproceedings{sokaravoiding,
  title={Avoiding Forgetting and Allowing Forward Transfer in Continual Learning via Sparse Networks},
  author={Sokar, Ghada and Mocanu, Decebal Constantin and Pechenizkiy, Mykola}
  booktitle={Joint European conference on machine learning and knowledge discovery in databases},
  year={2022},
  organization={Springer}
}
```
