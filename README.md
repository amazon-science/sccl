# Supporting Clustering with Contrastive Learning, NAACL 2021
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
    @inproceedings{
		zhang2021sccl,
		title={Supporting Clustering with Contrastive Learning},
		author={Dejiao Zhang, Feng Nan, Xiaokai Wei, Daniel Li, Henghui Zhu, Kathleen McKeown, Ramesh Nallapati, Andrew O. Arnold, Bing Xiang},
		booktitle={NAACL 2021},
		year={2021},
	}