import os
from transformers import BertForSequenceClassification, AutoModel, AutoConfig

import argparse

def save_bert(out_dir, base_name="prajjwal1/bert-mini", model_name='bert-mini', n_labels=5):
    
    save_file = os.path.join(out_dir, model_name + '_bert_params_' + str(n_labels))
    if not os.path.exists(save_file):
        config = AutoConfig.from_pretrained(base_name, 
                                            num_labels=n_labels,
                                            problem_type='regression',
                                            ignore_mismatched_sizes=True)
        
        bert = AutoModel.from_config(config)
        bert.save_pretrained(save_file)

if __name__ == "__main__":
    
    ###### define parser ######
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', 
                        help='Trains the transformer to predict flare types with given number of labels, values: 2, 3, 5(default)', 
                        nargs=1)
    args = parser.parse_args()
    
    if args.labels == None:
        NLABELS = 5
    else:
        NLABELS = int(args.labels[0])
        
    OUT_PATH = os.path.join(os.getcwd(), 'model_params')
        
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH, exist_ok=True)

    save_bert(OUT_PATH, n_labels=NLABELS)