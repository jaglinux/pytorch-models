import argparse
import sys

from transformers import AutoModelForCausalLM
import datasets

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

    def params(self):
        for k in self.model.state_dict().keys():
            print("parameter name is ", k)
            np = self.model.state_dict()[k].detach().numpy()
            print("shape is ", np.shape)
            print("numpy values are ", np)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="hf-models", description="List of models stored in HF, list model details, pre train it, fine tune etc")
    parser.add_argument("--model", help="Provide model name such as gpt2", default="gpt2")
    parser.add_argument("--print", help="Print model details", choices=("basic", "verbose"))
    parser.add_argument("--params", help="Print model parameters, weights and biases", action="store_true")
    args = parser.parse_args()
    if len(sys.argv) == 1:
        # just print gpt2 model
        model = model()
        model.print()
        sys.exit(0)
    model_param = args.model
    model = model(model_param)
    if args.print: model.print(args.print)
    if args.params: model.params()
