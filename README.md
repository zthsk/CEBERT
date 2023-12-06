
# Implementation of CEBERT
This is the official implementation of our paper [Robust Hate Speech Detection via Mitigating Spurious Correlations](https://aclanthology.org/2022.aacl-short.7) (Tiwari et al., AACL-IJCNLP 2022)


## Abstract
We develop a novel robust hate speech detection model that can defend against both word- and character-level adversarial attacks. We identify the essential factor that vanilla detection models are vulnerable to adversarial attacks is the spurious correlation between certain target words in the text and the prediction label. To mitigate such spurious correlation, we describe the process of hate speech detection by a causal graph. Then, we employ the causal strength to quantify the spurious correlation and formulate a regularized entropy loss function. We show that our method generalizes the backdoor adjustment technique in causal inference. Finally, the empirical evaluation shows the efficacy of our method.
## Dependencies
    python==3.8.10
    torch==1.11.0
    transformers==4.5.1
    argparse==1.1
## Arguements
Modify the following arguements as per your need.

    --n_augment  
    --lamda      
    --epoch      
    --batch_size 
    --model_type
For more information about the arguements, run the following command:
```bash
    python CEBERT.py -h
```
## Deployment

To train the baseBERT on default settings, run the following command:

```bash
    python CEBERT.py 
```

To train the CEBERT on default settings, run the following command:
```bash
    python CEBERT.py --model_type CEBERT
```

## Citation
```bash
@inproceedings{tiwari2022robust,
  title={Robust Hate Speech Detection via Mitigating Spurious Correlations},
  author={Tiwari, Kshitiz and Yuan, Shuhan and Zhang, Lu},
  booktitle={Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing},
  pages={51--56},
  year={2022}
}
```
