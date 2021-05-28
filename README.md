# Supporting Clustering with Contrastive Learning
[SCCL (NAACL 2021)](https://www.aclweb.org/anthology/2021.naacl-main.427.pdf) 
Dejiao Zhang, Feng Nan, Xiaokai Wei, Shangwen Li, Henghui Zhu,
Kathleen McKeown, Ramesh Nallapati, Andrew Arnold, and Bing Xiang. 


## Requirements

### Datasets:
  In additional to the original data, SCCL requires a pair of augmented data for each 
instance. See our paper for details. 

### Dependencies:
    python==3.6. 
    pytorch==1.6.0. 
    sentence-transformers==0.3.8. 
    transformers==3.3.0. 
    tensorboardX==2.1.  

## To run the code:
    1. put your dataset in the folder "./datasamples"  # for some license issue, we are not able to release the dataset now, we'll release the datasets asap
    2. bash ./scripts/run.sh # you need change the dataset info and results path accordingly


## Citation:
    @inproceedings{zhang-etal-2021-supporting,
    title = "Supporting Clustering with Contrastive Learning",
    author = "Zhang, Dejiao  and Nan, Feng  and Wei, Xiaokai  and
      Li, Shang-Wen  and Zhu, Henghui  and McKeown, Kathleen  and
      Nallapati, Ramesh  and Arnold, Andrew O.  and Xiang, Bing",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.427",
    pages = "5419--5430",
    abstract = " ",}