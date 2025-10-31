---
layout: post.html
title: SimCLR Model Implementation
tags: post
date: 2025-09-27T11:35:00
---
The architecture proposed in the **“A Simple Framework for Contrastive Learning of Visual Representations”** is named SimCLR and consists of the components below:
- `Encoder` (Using `ResNet` architecture)
- `ProjectionHead` (Simple MLP for producing the embeddings)
- `Augmenter` (Embedded into the `data.Dataset` object)

The loss for the architecture is `InfoNCE` (also known as `NT-Xent`) and is incorporated to the architecture.

# 01:Augmenter
The augmentation pipeline is required to generate 2 separate views of the same instance.

$$ \large
\tilde{x}\_i = t(x\_k) \rightarrow t \sim \mathcal{T}
$$

$$ \large
\tilde{x}\_j = t'(x\_k) \rightarrow t' \sim \mathcal{T}
$$

where:
- $\large x\_k$: Sampled minibatch $\large \{ x\_k \}\_{k=1}^{N}$
- $\large t$ and $\large t'$: Transformation functions (derived from the same $\large \mathcal{T}$ transformation set)
- $\large \tilde{x}\_i$ and $\large \tilde{x}\_j$: Two views of $\large x\_k$, described as **positive pairs**

```python
# --- Transforms
# A dataclass to store two separate transformations to use it for data augmentation
# -----------------------------------------------------------------------------------
@dataclass
class Transforms:
    BASE: T.Compose = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],      # ImageNet Mean
            std=[0.229, 0.224, 0.225]       # ImageNet Standart Deviation
        )
    ])
  
    AUGMENTATION: T.Compose = T.Compose([
        T.RandomResizedCrop(size=128),
        T.RandomHorizontalFlip(),
        T.RandomApply([
            T.ColorJitter(
                brightness=JitterParams.BRIGHTNESS,
                contrast=JitterParams.CONTRAST,
                saturation=JitterParams.SATURATION,
                hue=JitterParams.HUE
            ),
        ]),
        T.RandomGrayscale(p=.2),
        T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    ])
```

> [!INFO] Info
> The `Transforms.BASE` is used to **transform the images to tensors in appropriate format.** The `Transforms.AUGMENTATION` is used for augmentation, thus **generating the 2 views of the same instance.**

Augmentation is embedded into the `data.Dataset` object:

```python
# BioTrove(data.Dataset)
def __getitem__(self, index):
	"""
	Retrieves one data sample at the given index.
	"""
	path, label = self.data[index]
	
	img = Image.open(path, mode='r').convert('RGB')
	
	img = self._tbase(img)
	# Get pair of augmented views
	xi = self._taug(img)
	xj = self._taug(img)
	
	# Return the views and label pair
	return (xi, xj), torch.tensor(label, dtype=torch.long)
```
# 02:The Encoder
The encoder architecture is used to **generate embeddings from the positive pairs,** generated using the [[Model Architecture#01 Augmenter|Augmenter]] layer.

> [!INFO] Info
> The Encoder serves as the **embedding function.** It’s a ResNet architecture with ability to select from the configurations available. The last **fully-connected layer (fc)** is configured to match a specific dimension set with `_out_dims` parameter.

$$ \large
f\_\theta(x) : \mathcal{X} \rightarrow \mathbb{R}^{d}
$$
where:
- $\large f\_\theta$: Embedding function (a `ResNet` in this case)
- $\large d$: Embedding Dimension (Shape)

A `dataclass` named `Backbones` is implemented to hold the available ResNet architectures provided by the `torchvision` library.

```python
# --- ResNet Base Models
# A dataclass to hold references to different ResNet architectures
# This allows easy selection of a backbone model by name
# -----------------------------------------------------------------------------------
@dataclass
class Backbones:
    RESNET18 = models.resnet18
    RESNET34 = models.resnet34
    RESNET50 = models.resnet50
    RESNET101 = models.resnet101
    RESNET152 = models.resnet152
  
    @staticmethod
    def choose(_name: str, **kwargs) -> models.ResNet:
        """
        Return the ResNet model specified by '_name' with the final
        fully connected layer replaced.
        """
        # Map string names to model constructors
        map = {
            'resnet18': Backbones.RESNET18,
            'resnet34': Backbones.RESNET34,
            'resnet50': Backbones.RESNET50,
            'resnet101': Backbones.RESNET101,
            'resnet152': Backbones.RESNET152,
        }
  
        if _name not in map:
            raise ValueError(f"Model '{_name}' not recognized.\nAvailable models: {list(map.keys())}")
        model = map[_name](**kwargs)
        
        return model
  
 
# --- Base Encoder
# The Encoder class using a ResNet backbone
# The final fully connected layer is replaced to match the desired output dimension
# -----------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, _model: str, _out_dim: int, **kwargs) -> None:
        """
        Encoder model for feature extraction using a ResNet backbone.
        """
        super(Encoder, self).__init__()
        self.out_dim = _out_dim
        self.base = self._get_base_model(_model, self.out_dim, **kwargs)
        self.weights = kwargs.get('weights', None)
  
 
    def _get_base_model(self, _model: str, _out_dims, **kwargs) -> models.ResNet:
        """
        Return the base ResNet model (ResNet 18) with given
        configuration.
        """
        base_model = Backbones.choose(_model, **kwargs)
        # Replace the final fully connected layer with a new Linear layer
        base_model.fc = nn.Linear(base_model.fc.in_features, _out_dims)
        return base_model
  
    def forward(self, X) -> torch.Tensor:
        return self.base(X)
```
# 03:The Projection Head
...
# 04:Loss
The loss used in the training of the `SimCLR` implementation is known as **NTXent (Normalized Temperature-scaled Cross-Entropy) Loss** which is a known loss function for [[Contrastive Learning|contrastive learning]] task and proposed in the original paper.

> [!INFO] Purpose
> The core principle for the NTXent loss is simple, it **keeps positive pairs (similar samples) closer to each other** while **pushing negative samples as far away as possible** in the embedding space.
## 04.1:Positive/Negative Pairs
In contrastive learning, sample pairs are separated into 2 types; positive and negative.
- **Positive pairs ($\large z\_p$):** Different views[^1] of the same sample.
- **Negative pairs ($\large z\_k$):** Views of a different sample.

To measure how pairs of samples are positive/negative, a similarity metric is used which in this case is **cosine similarity:**

$$ \large
\text{sim}(\mathbf{z}\_a,\ \mathbf{z}\_i) = \mathbf{z}\_a^{\top} \mathbf{z}\_i
$$

Here:
- $\large \mathbf{z}\_a$: The anchor[^2]
- $\large \mathbf{z}\_i$: $\large i^{th}$ representation

$$ \large
\mathbf{z}\_i \in \mathcal{Z}
$$
Where:
- $\large \mathcal{Z}$: All of the representations over a batch of size $\large N$.

$$ \large
\mathcal{Z} = \{ \mathbf{z}\_1,\ \mathbf{z}\_2,\ \mathbf{z}\_3,\ \cdots,\ \mathbf{z}\_{2N} \}
$$

> [!WARNING] Assumption
> Here, we assume that the **representations are first $\large \mathscr{l}\_2$-normalized.**
> $$ \large
> \mathbf{z} \leftarrow \frac{\mathbf{z}}{|\mathbf{z}|\_2}
> $$

The NTXent loss for a single anchor $\large z\_i$​ is defined as the negative log of the probability of identifying its positive counterpart $\large z\_j$ from a set of 2N−2 negative samples