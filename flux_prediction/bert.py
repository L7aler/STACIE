import os
from transformers import BertForSequenceClassification,  AutoModel, AutoConfig

def save_bert(base_name="prajjwal1/bert-mini", model_name='bert-mini', param_name='bert_params', n_labels=5):
    
    save_file = os.path.join(os.getcwd(), 'model_params', model_name + '_' + param_name + '_' + str(n_labels))
    if not os.path.exists(save_file):
        config = AutoConfig.from_pretrained(base_name, 
                                            num_labels=n_labels,
                                            problem_type='regression',
                                            ignore_mismatched_sizes=True)
                                            #output_attentions = False, # Whether the model returns attentions weights.
                                            #output_hidden_states = False, # Whether the model returns all hidden-states.
                                            #force_download = True)
        """"
        bert = BertForSequenceClassification.from_pretrained(base_name,
                                                              num_labels=n_labels,
                                                              problem_type='regression',
                                                              ignore_mismatched_sizes=True)
        """
        
        bert = AutoModel.from_config(config)
        bert.save_pretrained(save_file)

save_bert(param_name='bert_params', n_labels=3)