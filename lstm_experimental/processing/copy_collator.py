import torch
import random
from transformers import DataCollatorForLanguageModeling

class DataCollatorWithUniformRandomOffsetsForCausalLM_copy(DataCollatorForLanguageModeling):
    """
    Data collator for causal language modeling with uniform random offsets for all sequences in a batch.
    Inherits from DataCollatorForLanguageModeling to handle tokenization and MLM if needed.
    """
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, max_offset=30):
        super().__init__(tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.max_offset = max_offset
        self.equal_token_id = tokenizer.convert_tokens_to_ids('=')[0]  # Token ID for '='
    
    def __call__(self, examples):
        # print(examples)
        inputs = torch.stack([example['input_ids'] for example in examples])
        labels = torch.stack([example['input_ids'] for example in examples])
        
        # Generate a uniform random offset to apply across the batch
        offset = random.randint(-self.max_offset, self.max_offset)
        # print(offset)
        # print(inputs)
        # Initialize lists for storing adjusted inputs and labels
        adjusted_inputs = []
        adjusted_labels = []

        # Apply the same offset to both inputs and labels
        for seq in inputs:
            shifted_input = seq[:len(seq) - offset] if offset > 0 else seq[-offset:]
            adjusted_inputs.append(shifted_input)
        
        for seq in labels:
            shifted_label = seq[:len(seq) - offset] if offset > 0 else seq[-offset:]
            adjusted_labels.append(shifted_label)

        # Stack adjusted inputs and labels into tensors
        adjusted_inputs_tensor = torch.stack(adjusted_inputs)
        adjusted_labels_tensor = torch.stack(adjusted_labels)

        # Prepare the tensor with the "=" token
        equal_token_tensor = torch.full((adjusted_inputs_tensor.size(0), 1), self.equal_token_id, dtype=torch.long)

        # Concatenate inputs, "=" token, and labels
        concatenated_inputs_labels = torch.cat([adjusted_inputs_tensor, equal_token_tensor, adjusted_labels_tensor], dim=1)
        #print(concatenated_inputs_labels)
        return {
            "input_ids": concatenated_inputs_labels,
            "attention_mask": (concatenated_inputs_labels != self.tokenizer.pad_token_id), #.long(),
            "labels": adjusted_labels_tensor
        }
              
# data_collator = DataCollatorWithUniformRandomOffsetsForCausalLM_copy(tokenizer, mlm=False, max_offset=30)
