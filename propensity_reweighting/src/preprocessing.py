from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

class ProScoreVectorizer():
    def __init__(self, config):
        """
        """
        self.config = config
        
        self.model = AutoModelForMaskedLM.from_pretrained(self.config["model"]) 
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer"])
        
    def vectorize_texts(self, texts_list):
        """
        """
        cls_matrix = np.zeros((len(texts_list), self.model.config.hidden_size))
        
        for j, sent in enumerate(texts_list):
            tokenized_sent = self.tokenizer.encode_plus(sent, 
                                               add_special_tokens = True, 
                                               truncation = True,  
                                               return_attention_mask=False, 
                                               return_token_type_ids=False, 
                                               return_tensors = 'pt')

            _cls = self.model.bert(tokenized_sent['input_ids']).last_hidden_state[:,0,:]
            cls_matrix[j, :] = _cls.detach().numpy()[0]
            
        return cls_matrix
    
    