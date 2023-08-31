# Awesome-Language-Model-on-Graphs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
A curated list of papers and resources about language model on graphs.

## Contents
- [Awesome-LM-on-Graphs](#awesome-lm-on-graphs)
  - [Contents](#contents)
  - [Datasets](#datasets)
  - [Representation Learning](#representation-learning)
  - [Pretraining](#pretraining)
  - [Distillation](#distillation)
  - [Classification](#classification)
  - [Contribution](#contribution)


## Datasets
### Text-attributed network
- Microsoft Academic network (MAG)
<br>Networks from 19 domains including CS, Mathematics, Geology, etc.
<br>Nodes: papers/authors/venues; Edges: citation/co-authorship/publish-in.
<br>Text: paper title, paper abstract on nodes.
<br>[[PDF](https://arxiv.org/abs/2302.03341)] [[Data](https://zenodo.org/record/7611544)] [[Preprocessing Code](https://github.com/PeterGriffinJin/Patton/blob/main/data_process/process_mag.ipynb)]
- Amazon Items
<br>Networks from 24 domains including Home, Clothing, Sports, etc.
<br>Nodes: items; Edges: co-purchase/co-viewed/same-brand.
<br>Text: item title and description on nodes.
<br>[[PDF](https://arxiv.org/pdf/1602.01585.pdf)] [[Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)] [[Preprocessing Code](https://github.com/PeterGriffinJin/Patton/blob/main/data_process/process_amazon.ipynb)]



## Representation Learning
- [GraphFormers: GNN-nested Transformers for
Representation Learning on Textual Graph](https://arxiv.org/abs/2105.02605)
<br>*NeurIPs 2021*.
<br>[[PDF](https://arxiv.org/abs/2105.02605)] [[Code](https://github.com/microsoft/GraphFormers)]
<!-- <br>Junhan Yang, Zheng Liu, Shitao Xiao, Chaozhuo Li, Defu Lian, Sanjay Agrawal, Amit Singh, Guangzhong Sun, Xing Xie. -->


- [Heterformer: Transformer-based Deep Node Representation Learning on Heterogeneous Text-Rich Networks](https://arxiv.org/abs/2205.10282)
<br>*KDD 2023*.
<br>[[PDF](https://arxiv.org/abs/2205.10282)] [[Code](https://github.com/PeterGriffinJin/Heterformer)]
<!-- <br>Bowen Jin, Yu Zhang, Qi Zhu, Jiawei Han. -->


- [Edgeformers: Graph-Empowered Transformers for Representation Learning on Textual-Edge Networks](https://openreview.net/pdf?id=2YQrqe4RNv)
<br>*ICLR 2023*.
<br>[[PDF](https://openreview.net/pdf?id=2YQrqe4RNv)] [[Code](https://github.com/PeterGriffinJin/Edgeformers)]
<!-- <br>Bowen Jin, Yu Zhang, Yu Meng, Jiawei Han. -->


## Pretraining
- [SPECTER: Document-level Representation Learning using Citation-informed Transformers](https://arxiv.org/abs/2004.07180)
<br>*ACL 2020*.
<br>[[PDF](https://arxiv.org/abs/2004.07180)]
<!-- <br>Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, Daniel S. Weld. -->


- [LinkBERT: Pretraining Language Models with Document Links](https://arxiv.org/pdf/2203.15827.pdf)
<br>*ACL 2022*.
<br>[[PDF](https://arxiv.org/pdf/2203.15827.pdf)] [[Code](https://github.com/michiyasunaga/LinkBERT)]
<!-- <br>Michihiro Yasunaga, Jure Leskovec, Percy Liang. -->


- [DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining](https://cs.stanford.edu/~myasu/papers/dragon_neurips22.pdf)
<br>*NeurIPs 2022*.
<br>[[PDF](https://cs.stanford.edu/~myasu/papers/dragon_neurips22.pdf)] [[Code](https://github.com/michiyasunaga/dragon)]
<!-- <br>Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D. Manning, Percy Liang, Jure Leskovec. -->


- [Patton: Language Model Pretraining on Text-rich Networks](https://arxiv.org/abs/2305.12268)
<br>*ACL 2023*.
<br>[[PDF](https://arxiv.org/abs/2305.12268)] [[Code](https://github.com/PeterGriffinJin/Patton)]
<!-- <br>Bowen Jin, Wentao Zhang, Yu Zhang, Yu Meng, Xinyang Zhang, Qi Zhu, Jiawei Han. -->


- [Graph-Aware Language Model Pre-Training on a Large Graph
Corpus Can Help Multiple Graph Applications](https://arxiv.org/pdf/2306.02592.pdf)
<br>*KDD 2023*.
<br>[[PDF](https://arxiv.org/pdf/2306.02592.pdf)]
<!-- <br>Han Xie, Da Zheng, Jun Ma, Houyu Zhang, Vassilis N. Ioannidis, Xiang Song, Qing Ping, Sheng Wang, Carl Yang, Yi Xu, Belinda Zeng, Trishul Chilimbi. -->



## Efficiency & Model distillation
- [Train Your Own GNN Teacher: Graph-Aware Distillation on Textual Graphs](https://arxiv.org/abs/2304.10668)
<br>*PKDD 2023*.
<br>[[PDF](https://arxiv.org/abs/2304.10668)]
<!-- <br>C. Mavromatis, V. N. Ioannidis, S. Wang, D. Zheng, S. Adeshina, J. Ma, H. Zhao, C. Faloutsos, G. Karypis. -->


- [Efficient and effective training of language and graph neural network models](https://arxiv.org/abs/2206.10781)
<br>*AAAI 2023*.
<br>[[PDF](https://arxiv.org/abs/2206.10781)] [[Code]()]
<!-- <br>Vassilis N Ioannidis, Xiang Song, Da Zheng, Houyu Zhang, Jun Ma, Yi Xu, Belinda Zeng, Trishul Chilimbi, George Karypis. -->


## Classification
- [Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification](https://yuzhimanhua.github.io/papers/www22zhang.pdf)
<br>*WWW 2022*.
<br>[[PDF](https://yuzhimanhua.github.io/papers/www22zhang.pdf)] [[Code](https://github.com/yuzhimanhua/MICoL)]
<!-- <br>Yu Zhang, Zhihong Shen, Chieh-Han Wu, Boya Xie, Junheng Hao, Ye-Yi Wang, Kuansan Wang, Jiawei Han. -->


- [Learning on Large-scale Text-attributed graphs via variational inference](https://openreview.net/pdf?id=q0nmYciuuZN)
<br>*ACL 2023*.
<br>[[PDF](https://openreview.net/pdf?id=q0nmYciuuZN)] [[Code](https://github.com/AndyJZhao/GLEM)]
<!-- <br>Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, Jian Tang. -->

## Question Answering
- [GreaseLM: Graph Reasoning Enhanced Language Models for Question Answering](https://cs.stanford.edu/~myasu/papers/greaselm_iclr22.pdf)
<br>*ICLR 2022*.
<br>[[PDF](https://cs.stanford.edu/~myasu/papers/greaselm_iclr22.pdf)] [[Code](https://github.com/snap-stanford/GreaseLM)]
<!-- <br>Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D Manning and Jure Leskovec. -->


## Contribution
Contributions to this repository are welcome!

If you find any error or have relevant resources, feel free to open an issue or a pull request.


<!-- - []()
<br>
<br>**.
<br>[[PDF]()] [[Code]()] -->

<!-- add specter -->
