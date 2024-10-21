import torch
import random
from transformers import DataCollatorForLanguageModeling

class DataCollatorWithUniformRandomOffsetsForCausalLM_reverse(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, max_offset=30):
        super().__init__(tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.max_offset = max_offset
        self.equal_token_id = tokenizer.convert_tokens_to_ids('=')  # Token ID for '='
    
    def __call__(self, examples):
        inputs = torch.stack([example['input_ids'] for example in examples])
        
        # Generate a uniform random offset to apply across the batch
        offset = random.randint(-self.max_offset, self.max_offset)
        
        # Initialize lists for storing adjusted inputs and reversed labels
        adjusted_inputs = []
        reversed_labels = []

        # Apply the same offset to inputs and reverse them for labels
        for seq in inputs:
            shifted_input = seq[:len(seq) - offset] if offset > 0 else seq[-offset:]
            adjusted_inputs.append(shifted_input)
            reversed_labels.append(torch.flip(shifted_input, dims=[0]))  # Reverse the input for labels
            #print(adjusted_inputs[-1], reversed_labels[-1])
        # Stack adjusted inputs and reversed labels into tensors
        adjusted_inputs_tensor = torch.stack(adjusted_inputs)
        reversed_labels_tensor = torch.stack(reversed_labels)

        # Prepare the tensor with the "=" token
        equal_token_tensor = torch.full((adjusted_inputs_tensor.size(0), 1), self.equal_token_id, dtype=torch.long)

        # Concatenate inputs, "=" token, and labels
        concatenated_inputs = torch.cat([adjusted_inputs_tensor, equal_token_tensor], dim=1)
        concatenated_labels = torch.cat([equal_token_tensor, reversed_labels_tensor], dim=1)

        return {
            "input_ids": concatenated_inputs,
            "attention_mask": (concatenated_inputs != self.tokenizer.pad_token_id).long(),
            "labels": concatenated_labels  # Labels are reversed inputs
        }

# Example usage
data_collator = DataCollatorWithUniformRandomOffsetsForCausalLM_reverse(tokenizer, mlm=False, max_offset=30)
