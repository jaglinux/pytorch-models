import argparse
import sys

from transformers import AutoModelForCausalLM
import datasets

# model_name = "gpt2"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# print("model is ", model.state_dict().keys())
# print("model is ", model.state_dict()['transformer.wte.weight'].detach().numpy().shape)
class model:
    def __init__(self, name="gpt2"):
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(name)
        
    def print(self):
        print(f" {self.model} ")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="hf-models", description="List of models stored in HF, list model details, pre train it, fine tune etc")
    parser.add_argument("--model", help="Provide model name such as gpt2", default="gpt2")
    args = parser.parse_args()
    if len(sys.argv) == 1:
        # just print gpt2 model
        model = model()
        model.print()
        
    

