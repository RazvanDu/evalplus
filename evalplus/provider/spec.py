from typing import List

import sys
import os

cache_directory = os.getenv("cache_dir")
custom_path = os.getenv("copyspec_path")

if custom_path not in sys.path:
    sys.path.append(custom_path)

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from speculative_copying import SpeculativeDecoder
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)

class SpeculativeDecoderProvider(DecoderBase):
    def __init__(
        self,
        name: str,
        secondary_model: str = "",
        dataset: str = "",
        force_base_prompt: bool = False,
        instruction_prefix: str = "",
        response_prefix: str = "",
        device_map: str = "auto",
        dtype: str = "float16",
        max_new_tokens: int = 300,
        gamma: int = 5,
        **kwargs,
    ):
        """
        A speculative decoding provider for evalplus.

        Parameters:
            name (str): Primary model name.
            secondary_model (str): Secondary model used for speculative decoding.
            dataset (str): Dataset name.
            force_base_prompt (bool): Force base prompt if True.
            instruction_prefix (str): Instruction prefix for chat formatting.
            response_prefix (str): Response prefix for chat formatting.
            device_map (str): Device map for distributed inference.
            dtype (str): Torch dtype for model precision.
            max_new_tokens (int): Maximum tokens to generate.
        """
        super().__init__(name=name, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.force_base_prompt = force_base_prompt
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix
        self.total_copy = 0
        self.gamma = gamma

        print(f"[DEBUG] Initializing SpeculativeDecoderProvider for model: {name} on device: {self.device}")

        self.decoder = SpeculativeDecoder(
            target_model_name=name,
            draft_model_name=secondary_model,
            device=self.device,
        )
        print("[DEBUG] SpeculativeDecoder initialized successfully.")

        # Tokenizer and EOS settings

        self.tokenizer = self.decoder.tokenizer
        
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            self.eos += ["\n```\n"]

        self.total_tokens = 0

    def is_direct_completion(self) -> bool:
        """
        Check if the task requires direct completion (without chat formatting).
        """
        return self.force_base_prompt or getattr(self.tokenizer, "chat_template", None) is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 1,
    ) -> List[str]:
        """
        Generate code using speculative decoding.

        Parameters:
            prompt (str): Input prompt for code generation.
            do_sample (bool): Enable sampling if True.
            num_samples (int): Number of samples to generate (only supports 1 for now).

        Returns:
            List[str]: List of generated code outputs.
        """

        if num_samples > 1:
            raise ValueError("[ERROR] SpeculativeDecoderProvider only supports num_samples=1 currently.")

        # Prepare the prompt
        formatted_prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )
        print("PROMPT:", formatted_prompt)

        # Generate using speculative decoding
        try:

                
            input_tokens = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(
                self.device
            )

            generated_text, accepted_tokens = self.decoder.generate_raw(
                formatted_prompt,
                temperature=0.0,
                top_k=0,
                top_p=1,
                gamma=self.gamma,
                max_new_tokens=1024,
            )
            generated_text = generated_text[:, input_tokens.size(-1) :]

            self.total_tokens += len(generated_text[0])
            
            #print(self.total_tokens, "tokens generated thus far!")
            
            generated_text = self.tokenizer.batch_decode(
                generated_text,
                skip_special_tokens=self.skip_special_tokens,
            )[0]

            self.total_copy += accepted_tokens

        except Exception as e:
            print(f"[ERROR] SpeculativeDecoderProvider.codegen failed: {e}")
            return []

        min_index = 10000
        for eos in self.eos:
            if eos in generated_text:
                min_index = min(min_index, generated_text.index(eos))
        cleaned_output = generated_text[:min_index].replace("\t", "    ")

        print(f"[DEBUG] TOTALLLLL Tokens accepted: {self.total_copy}")
        return [cleaned_output]