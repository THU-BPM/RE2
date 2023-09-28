import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import logging, BertModel, AutoTokenizer, BertForSequenceClassification

# count = 0
def GetScore(m, s, r, k, lam, ds="semeval"):
    """
    Calculate the score based on the binary mask, scores and edge scores

    :param m: torch.Tensor, binary mask with shape (batch_size, sequence_length)
    :param s: torch.Tensor, scores with shape (batch_size, sequence_length)
    :param r: int, edge score
    :param k: int, maximum allowed mask selection

    :return: torch.Tensor, score with shape (batch_size, 1)
    """
    score = torch.sum(m * s * 2988.0, dim=-1)
    score += torch.sum(m[:, :-1] * m[:, 1:] * r, dim=-1)
    selected_nums = torch.cumsum(m, dim=-1)[..., -1:]
    lambda_multiplier = lam * (k - selected_nums)
    lambda_multiplier = lambda_multiplier.squeeze(-1)
    score += lambda_multiplier
    return score.unsqueeze(-1)

class GibbsSampler(nn.Module):
    """
    A Gibbs distribution sampler.
    """

    def __init__(self, device, in_features, perturb_scale=0.01, r=1, k=1, ds="semeval"):
        super(GibbsSampler, self).__init__()
        self.device = device
        self.training = True
        self.perturb_scale = perturb_scale
        self.r = r
        self.k = k
        self.ds = ds

    def train(self, mode=True):
        super().train(mode)
        self.training = mode

    def eval(self, mode=True):
        super().train(mode=not mode)
        self.training = not mode

    def forward(self, attention_mask, s, epsilon=None):
        """
        :param s: relation Scores

        :return: distribution, use z = dist.sample((sample_size,)) to get desired samples
        """

        init = torch.randint(0, 2, size=[s.shape[0], s.shape[1]], dtype=torch.int64).to(self.device)
        iter = init
        for i in range(0, self.r):
            for timestep in range(0, iter.shape[1], self.k):
                x = torch.cat((iter[..., :timestep], iter[..., timestep + 1 :]), dim=-1)
                index = torch.full((iter.shape[0], 1), timestep).to(self.device)
                combine = torch.cat([index, x, s], dim=-1)

                timestep_tensor = torch.tensor(timestep).to(self.device)
                if True:
                    iter_zero = iter.clone()
                    iter_zero[..., timestep] = 0
                    score_zero = GetScore(iter_zero * attention_mask, s, 300, 28, 510.3, ds=self.ds)
                    iter_one = iter.clone()
                    iter_one[..., timestep] = 1
                    score_one = GetScore(iter_one * attention_mask, s, 300, 28, 510.3, ds=self.ds)
                    score_all = torch.cat((score_zero, score_one), dim=-1)
                    prob_real = F.softmax(score_all, dim=-1)
                    # perturb-and-map with imported library
                    # import torch.distributions as dist
                    # prob_real = prob_real + dist.Normal(0, self.perturb_scale).sample(prob_real.shape).to(self.device)
                    # prob_real = prob_real / torch.sum(prob_real, dim=-1, keepdim=True)

                    z = torch.distributions.Categorical(probs=prob_real).sample()
                    iter[torch.arange(iter.shape[0]), timestep] = torch.gather(
                        torch.tensor([0, 1]).to(self.device), 0, z
                    )
                
                if timestep % 20 == 0:
                    print(
                        f"Iteration {i:4d} / {self.r:4d}, Step {timestep:4d} / {iter.shape[1]:4d} of Gibbs Sampling",
                        end='\r'
                    )
        return iter


class REModule():
    """
    Relation Extraction Module containing the BERT model and the Gibbs Sampler
    """
    def __init__(self, device, r=8, k=1, enc_size=128, num_classes=19, ds="semeval"):
        super(REModule, self).__init__()
        # disable the weights not used warning
        logging.set_verbosity_warning()
        logging.set_verbosity_error()
        self.device = device
        # Load the pre-trained BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        # Load the pre-trained BERT model
        self.bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes, output_hidden_states=True).to(self.device)
        # add special tokens
        self.bert_model.resize_token_embeddings(self.bert_model.get_input_embeddings().num_embeddings + 4)
        # what stands for <e1>, </e1>, <e2>, and </e2>?
        self.e1_l = self.tokenizer.convert_tokens_to_ids("<e1>")
        self.e1_r = self.tokenizer.convert_tokens_to_ids("</e1>")
        self.e2_l = self.tokenizer.convert_tokens_to_ids("<e2>")
        self.e2_r = self.tokenizer.convert_tokens_to_ids("</e2>")
        # Set Hyperparams
        self.r = r # Gibbs sampling iterations
        self.k = k # Gibbs sampling steps
        # Set encoding size
        self.enc_size = enc_size
        self.bert_depth = 768
        self.z_layer = GibbsSampler(self.device, enc_size, r=self.r, k=self.k, ds=ds)
        self.training = True
        self.sel_exceed_punishment = 0

    def train(self, mode=True):
        self.z_layer.train()
        self.bert_model.train()
        self.training = mode

    def eval(self, mode=True):
        self.z_layer.eval()
        self.bert_model.eval()
        self.training = not mode

    # load state dict
    def load_state_dict(self, state_dict):
        self.z_layer.load_state_dict(state_dict)
    
    # for saving model
    def state_dict(self):
        return self.z_layer.state_dict()

    def get_binary_mask(self, texts):
        """
        Calculates a binary mask

        :param texts: a batch of text
        :param entity1s: a batch of (start, end)
        :param entity2s: a batch of (start, end)

        :return: binary mask; last hidden state in testing mode, which will be used for evidence extraction
        """
        input_ids_list = []
        attention_mask_list = []
        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.enc_size,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            # get attention mask
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        input_ids_list = torch.cat(input_ids_list, dim=0)
        attention_mask_list = torch.cat(attention_mask_list, dim=0)
        outputs = self.bert_model(input_ids_list, attention_mask=attention_mask_list)[-1]
        last_hidden_state = outputs[-1]
        
        entity1_pos = [torch.where(input_ids_list[i] == self.e1_l)[0].item() if len(torch.where(input_ids_list[i] == self.e1_l)[0]) > 0 else 0 for i in range(len(texts))]
        entity1_end_pos = [torch.where(input_ids_list[i] == self.e1_r)[0].item() if len(torch.where(input_ids_list[i] == self.e1_r)[0]) > 0 else 0 for i in range(len(texts))]
        entity2_pos = [torch.where(input_ids_list[i] == self.e2_l)[0].item() if len(torch.where(input_ids_list[i] == self.e2_l)[0]) > 0 else 0 for i in range(len(texts))]
        entity2_end_pos = [torch.where(input_ids_list[i] == self.e2_r)[0].item() if len(torch.where(input_ids_list[i] == self.e2_r)[0]) > 0 else 0 for i in range(len(texts))]
        
        entity1_embs = [last_hidden_state[i][entity1_pos[i]] for i in range(len(texts))]
        entity1_embs = torch.stack(entity1_embs, dim=0)
        entity2_embs = [last_hidden_state[i][entity2_pos[i]] for i in range(len(texts))]
        entity2_embs = torch.stack(entity2_embs, dim=0)
        relation_extraction_hidden_state = (
            entity1_embs + entity2_embs
        )
        relation_scores = torch.bmm(last_hidden_state, relation_extraction_hidden_state.unsqueeze(2)).squeeze(-1)
        # normalize with L2 norm
        relation_scores = relation_scores / torch.norm(relation_scores, dim=1).unsqueeze(1)
        relation_scores = relation_scores * attention_mask_list
        masks_prob = self.z_layer.forward(attention_mask_list, relation_scores)
        for ind in range(masks_prob.shape[0]):
            masks_prob[ind, entity1_pos[ind]:entity1_end_pos[ind]+1] = 1
            masks_prob[ind, entity2_pos[ind]:entity2_end_pos[ind]+1] = 1
        if self.training:
            return masks_prob
        else:
            return masks_prob, last_hidden_state
    
    def decode_text(self, tokens, masks):
        """
        Decodes a batch of tokens

        :param tokens: a batch of tokens
        :param masks: a batch of masks

        :return: a batch of decoded text
        """
        texts = []
        for token, mask in zip(tokens, masks):
            text = self.tokenizer.decode(token[mask == 1])
            texts.append(text)
        return texts
    
    def bert_tokenize_only(self, texts):
        """
        Tokenizes a batch of texts

        :param texts: a batch of texts

        :return: a batch of tokens
        """
        tokens = []
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        for text in texts:
            token = self.tokenizer.encode(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.enc_size,
                truncation=True,
                pad_to_max_length=True,
                add_special_tokens=True
            ).to(self.device)
            tokens.append(token)
            
        tokens = torch.cat(tokens, dim=0)
        return tokens
