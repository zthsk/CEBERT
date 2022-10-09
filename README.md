Implementation of CEBERT


Dependencies
    python==3.8.10
    torch==1.11.0
    transformers==4.5.1
    argparse==1.1

Arguements
    --n_augment  
    --lamda      
    --epoch      
    --batch_size 
    --model_type 

Run: python CEBERT.py -h for more information about the arguements

To train the baseBERT on default settings, run the following command:
python CEBERT.py 

To train the CEBERT on default settings, run the following command:
python CEVERT.py --model_type CEBERT

