import glob
import json
import os
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class ModelProvider(ABC):
    @abstractmethod
    def evaluate_model(self, prompt: str) -> str: ...

    @abstractmethod
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]: ...

    @abstractmethod
    def encode_text_to_tokens(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str: ...
        

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 model_to_test: ModelProvider = None,
                 needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PaulGrahamEssays"),
                 retrieval_question = "What is the best thing to do in San Francisco?",
                 final_context_length_buffer = 200
                 ):
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.final_context_length_buffer = final_context_length_buffer
        self.model_to_test = model_to_test

    def evaluate(self, context_length, depth_percent):
        context = self.generate_context(context_length, depth_percent)
        prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)
        response = self.model_to_test.evaluate_model(prompt, self.needle)
        return response

    def generate_context(self, context_length, depth_percent):
        context = self.read_context_files(context_length)
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needle(context, depth_percent, context_length)
        return context
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.model_to_test.encode_text_to_tokens(self.needle)
        tokens_context = self.model_to_test.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.model_to_test.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.model_to_test.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.model_to_test.encode_text_to_tokens(context))

    def read_context_files(self, context_length):
        context = ""
        while self.get_context_length_in_tokens(context) < context_length:
            for file in glob.glob(os.path.join(self.haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def encode_and_trim(self, context, context_length):
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context

