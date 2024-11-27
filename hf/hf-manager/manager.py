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
        
    def print(self, choice):
        if choice == "basic":
            print(f" {self.model} ")
        elif choice == "verbose":
            print(f" {self.model.__dict__}")
        else:
            print("ERROR - model print : wrong choice")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="hf-models", description="List of models stored in HF, list model details, pre train it, fine tune etc")
    parser.add_argument("--model", help="Provide model name such as gpt2", default="gpt2")
    parser.add_argument("--print", help="Print model details", choices=("basic", "verbose"))
    args = parser.parse_args()
    if len(sys.argv) == 1:
        # just print gpt2 model
        model = model()
        model.print()
        sys.exit(0)
    model_param = args.model
    model = model(model_param)
    if args.print: model.print(args.print)
