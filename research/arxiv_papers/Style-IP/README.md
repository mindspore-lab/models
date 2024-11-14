# 一种新的边缘计算模型访问控制和隐私增强方法

# A Novel Access Control and Privacy-Enhancing Approach for Models in Edge Computing

In this paper, we propose a novel intrinsic access control scheme tailored for edge computing, utilizing image style as a licensing mechanism. This approach innovatively integrates image style recognition into the model’s internal operational framework, enabling the model to only perform valid inferences on inputs that match specific styles, while invalidating forged styles and arbitrary images. By restricting the model's input data, our method achieves proactive protection of intellectual property, and since the original data must undergo style transfer during use, it also enhances data privacy to a certain extent. Compared to traditional methods, our approach offers greater flexibility and adaptability, making it better suited to address the complex threats in edge computing environments. Extensive experiments conducted on benchmark datasets such as MNIST, CIFAR-10, and FaceScrub demonstrate that our scheme excels in usability, security, and robustness, showcasing its broad application prospects in edge computing scenarios.


Keywords：Edge computing, Access control, Privacy enhancement, Intellectual property protection, AI-based cybersecurity services.

This paper has been published in [arXiv](https://arxiv.org/abs/2411.03847).

#### Test Environment:  

```markdown
- MindSpore 2.2.14  
- Python 3.8.3
- Cuda 11.1  
- RTX 3080x2 
```

#### Usage：  

install MindSpore
```bash  
mindspore==2.2.14
```

The Style transfer model is trained with Style***.py code, and the original data set is processed to generate the style transfer data set

```bash
python Style***.py 
```

For the model of ResNet architecture, a secure deep learning model based on style transfer image for access control is obtained by using the corresponding style transfer data set for training.

```bash
python ResNet***.py 
```


#### Thank you to the MindSpore community for your support!