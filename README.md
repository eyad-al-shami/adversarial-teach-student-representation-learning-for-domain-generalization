# adversarial-teach-student-representation-learning-for-domain-generalization
Implementation of the paper "Adversarial Teach Student Representation Learning for Domain Generalization"

Getting Started
To start training, you can choose one of the predefined experiments or create your own experiment by overriding the default settings. 
To run the code, use the following command:

```python:
python train.py --use_cuda --experiment_cfg "Experiments/resnet_18.yaml" --logger wandb --experiment_name "lr"
```

Note that for now, the --logger wandb option is mandatory, as we are currently working on facilitating other types of loggers.

Acknowledgments
This implementation is based on the paper "Adversarial Teach Student Representation Learning for Domain Generalization", by X. Peng, Q. Yao, and D. Cosker. We would like to thank the authors for their contribution to the field.

License
This project is licensed under the MIT License - see the LICENSE file for details.
