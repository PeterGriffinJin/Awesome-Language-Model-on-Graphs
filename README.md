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
  - [Graph As Tools](#classification)
  - [Contribution](#contribution)

### Keywords Convention

![](https://img.shields.io/badge/EncoderOnly-blue) The transformer architecture used in the work, e.g., EncoderOnly, DecoderOnly, EncoderDecoder.

![](https://img.shields.io/badge/Medium-red) The size of the language model, e.g., medium, LLM.


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

1. **SPECTER: Document-level Representation Learning using Citation-informed Transformers.** `ACL 2020`

    *Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, Daniel S. Weld.* [[Paper](https://arxiv.org/abs/2004.07180)], 2020.4, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph.** `NeurIPs 2021`

    *Junhan Yang, Zheng Liu, Shitao Xiao, Chaozhuo Li, Defu Lian, Sanjay Agrawal, Amit Singh, Guangzhong Sun, Xing Xie.* [[Paper](https://arxiv.org/abs/2105.02605)][[Code]](https://github.com/microsoft/GraphFormers), 2021.5, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings.** `EMNLP 2022`

    *Junhan Yang, Zheng Liu, Shitao Xiao, Chaozhuo Li, Defu Lian, Sanjay Agrawal, Amit Singh, Guangzhong Sun, Xing Xie.* [[Paper](https://arxiv.org/pdf/2202.06671.pdf)][[Code]](https://github.com/malteos/scincl), 2022.2, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Heterformer: Transformer-based Deep Node Representation Learning on Heterogeneous Text-Rich Networks.** `KDD 2023`

    *Bowen Jin, Yu Zhang, Qi Zhu, Jiawei Han.* [[Paper](https://arxiv.org/abs/2205.10282)][[Code]](https://github.com/PeterGriffinJin/Heterformer), 2022.5, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Edgeformers: Graph-Empowered Transformers for Representation Learning on Textual-Edge Networks.** `ICLR 2023`

    *Bowen Jin, Yu Zhang, Yu Meng, Jiawei Han.* [[Paper](https://openreview.net/pdf?id=2YQrqe4RNv)][[Code]](https://github.com/PeterGriffinJin/Edgeformers), 2023.1, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)


## Pretraining

1. **LinkBERT: Pretraining Language Models with Document Links.** `ACL 2022`

    *Michihiro Yasunaga, Jure Leskovec, Percy Liang.* [[Paper](https://arxiv.org/pdf/2203.15827.pdf)][[Code]](https://github.com/michiyasunaga/LinkBERT), 2022.3, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining.** `NeurIPs 2022`

    *Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D. Manning, Percy Liang, Jure Leskovec.* [[Paper](https://cs.stanford.edu/~myasu/papers/dragon_neurips22.pdf)][[Code]](https://github.com/michiyasunaga/dragon), 2022.10, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Patton: Language Model Pretraining on Text-rich Networks.** `ACL 2023`

    *Bowen Jin, Wentao Zhang, Yu Zhang, Yu Meng, Xinyang Zhang, Qi Zhu, Jiawei Han.* [[Paper](https://arxiv.org/abs/2305.12268)][[Code]](https://github.com/PeterGriffinJin/Patton), 2023.5, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Graph-Aware Language Model Pre-Training on a Large Graph Corpus Can Help Multiple Graph Applications.** `KDD 2023`

    *Han Xie, Da Zheng, Jun Ma, Houyu Zhang, Vassilis N. Ioannidis, Xiang Song, Qing Ping, Sheng Wang, Carl Yang, Yi Xu, Belinda Zeng, Trishul Chilimbi.* [[Paper](https://arxiv.org/pdf/2306.02592.pdf)], 2023.6, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)


## Efficiency & Model distillation

1. **Efficient and effective training of language and graph neural network models.** `AAAI 2023`

    *Vassilis N Ioannidis, Xiang Song, Da Zheng, Houyu Zhang, Jun Ma, Yi Xu, Belinda Zeng, Trishul Chilimbi, George Karypis.* [[Paper](https://arxiv.org/abs/2206.10781)], 2022.6, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Train Your Own GNN Teacher: Graph-Aware Distillation on Textual Graphs.** `PKDD 2023`

    *C. Mavromatis, V. N. Ioannidis, S. Wang, D. Zheng, S. Adeshina, J. Ma, H. Zhao, C. Faloutsos, G. Karypis.* [[Paper](https://arxiv.org/abs/2304.10668)], 2023.4, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)


## Classification

1. **Node Feature Extraction by Self-Supervised Multi-Scale Neighborhood Prediction.** `ICLR 2022`

    *Eli Chien, Wei-Cheng Chang, Cho-Jui Hsieh, Hsiang-Fu Yu, Jiong Zhang, Olgica Milenkovic, Inderjit S Dhillon.* [[Paper](https://arxiv.org/pdf/2111.00064.pdf)][[Code](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt)], 2021.11, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification.** `WWW 2022`

    *Yu Zhang, Zhihong Shen, Chieh-Han Wu, Boya Xie, Junheng Hao, Ye-Yi Wang, Kuansan Wang, Jiawei Han.* [[Paper](https://yuzhimanhua.github.io/papers/www22zhang.pdf)][[Code](https://github.com/yuzhimanhua/MICoL)], 2022.2, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Learning on Large-scale Text-attributed graphs via variational inference.** `ICLR 2023`

    *Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, Jian Tang.* [[Paper](https://openreview.net/pdf?id=q0nmYciuuZN)][[Code](https://github.com/AndyJZhao/GLEM)], 2023.1, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

1. **Explanations as Features: LLM-Based Features for Text-Attributed Graphs.** `preprint`

    *Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, Bryan Hooi.* [[PDF](https://arxiv.org/pdf/2305.19523.pdf)] [[Code](https://github.com/XiaoxinHe/TAPE)], 2023.5, ![](https://img.shields.io/badge/EncoderOnly-blue)![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/Medium-red) ![](https://img.shields.io/badge/LLM-red)

1. **Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs.** `preprint`

    *Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Hongzhi Wen, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui Liu, Jiliang Tang.* [[PDF](https://arxiv.org/abs/2307.03393)] [[Code](https://github.com/CurryTang/Graph-LLM)], 2023.7, ![](https://img.shields.io/badge/EncoderOnly-blue)![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/Medium-red) ![](https://img.shields.io/badge/LLM-red)

1. **Natural Language is All a Graph Needs.** `preprint`

    *Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, Yongfeng Zhang.* [[PDF](https://arxiv.org/abs/2308.07134)], 2023.8, ![](https://img.shields.io/badge/DecoderOnly-blue)![](https://img.shields.io/badge/EncoderDecoder-blue) ![](https://img.shields.io/badge/LLM-red)

1. **GraphText: Graph Reasoning in Text Space.** `preprint`

    *Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, Jian Tang.* [[PDF](https://arxiv.org/abs/2310.01089)], 2023.10, ![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/LLM-red)


## Question Answering

1. **GreaseLM: Graph Reasoning Enhanced Language Models for Question Answering.** `ICLR 2022`

    *Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D Manning and Jure Leskovec.* [[PDF](https://cs.stanford.edu/~myasu/papers/greaselm_iclr22.pdf)] [[Code](https://github.com/snap-stanford/GreaseLM)], 2022.1, ![](https://img.shields.io/badge/EncoderOnly-blue) ![](https://img.shields.io/badge/Medium-red)

## Graph As Tools
1. **Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT.** `preprint`

    *Jiawei Zhang.* [[PDF](https://arxiv.org/abs/2304.11116)] [[Code](https://github.com/jwzhanggy/Graph_Toolformer)], 2023.4, ![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/LLM-red)

1. **StructGPT: A General Framework for Large Language Model to Reason over Structured Data.** `preprint`

    *Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, Ji-Rong Wen.* [[PDF](https://arxiv.org/abs/2305.09645)] [[Code](https://github.com/RUCAIBox/StructGPT)], 2023.5, ![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/LLM-red)




## Contribution
Contributions to this repository are welcome!

If you find any error or have relevant resources, feel free to open an issue or a pull request.


<!-- 1. **xxx.** `xxx 2022`

    *xxx.* [[PDF]()] [[Code]()], 2022.1 -->
