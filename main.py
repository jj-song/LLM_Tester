import torch
import transformers
from transformers import AutoTokenizer


def run_model():
    name = 'mosaicml/mpt-7b-instruct'

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'torch'
    config.init_device = 'cuda:0'  # For fast initialization directly on GPU!
    #config.max_seq_len = 4096  # (input + output) tokens can now be up to 4096

    model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)

    # Define the prompt text
    prompt_text = "Write a unit test for the C# method: "

    # Read the contents from the "level_1" text file and append to the prompt
    with open('methods\\level_1', 'r') as file:
        prompt_text += file.read()

    # Tokenize the input

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to('cuda:0')
    #input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to('cuda:0')

    # Generate a response
    # output_ids = model.generate(input_ids)
    output_ids = model.generate(input_ids, max_new_tokens=150)

    # Decode the response
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("Response Output: ")
    print(output_text)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
