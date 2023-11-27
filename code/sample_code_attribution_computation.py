from transformers import BertTokenizer
# from .. import Transformer_Explainability
from Transformer_Explainability.BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from Transformer_Explainability.BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification

def compute_att(model, tokenizer, text):
    explanations = Generator(model)
    explanations_orig_lrp = Generator(model)
    
    method_expl = {"transformer_attribution": explanations.generate_LRP,
                        "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
                        "last_attn": explanations_orig_lrp.generate_attn_last_layer,
                        "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
                        "lrp": explanations_orig_lrp.generate_full_lrp,
                        "rollout": explanations_orig_lrp.generate_rollout}    
    all_att = []
    all_token = []
    attribute_method = method_expl["partial_lrp"]
    encoding = tokenizer(text, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to("cuda")
    attention_mask = encoding['attention_mask'].to("cuda")
    expl = attribute_method(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    # normalize scores
    expl = expl[1:-1]
    # expl = (expl - expl.min()) / (expl.max() - expl.min())
    expl = expl.detach().cpu().numpy()
    input_ids_cpu = input_ids.flatten().detach().cpu().numpy()[1:-1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu)

    assert len(expl) == len(tokens)
    all_att.append(expl)
    all_token.append(tokens)
        
    return all_att, all_token

model_hook = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_path)
model_hook = model_hook.to(device)
all_att_test, all_token_test = compute_att(model_hook, tokenizer, clean_test_text_list)

