from transformers import GPT2Tokenizer
from opt.opt_model import OPTModel,OPTForCausalLM
import torch

tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
origin_model = OPTModel.from_pretrained("facebook/opt-350m")
gen_model=OPTForCausalLM.from_pretrained('facebook/opt-350m')
inputs = tokenizer("Hello,my dog is cute", return_tensors="pt")
print(gen_model.generate(inputs))