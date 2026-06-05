import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalChatbot:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None):
        self.model_name = model_name
        
        # Select device automatically
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model {model_name} on {self.device}...")
        # For CPU/MPS, we load in float32 or bfloat16/float16 depending on support
        if self.device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=self.device
            )
        else:
            # For GPU or MPS
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "mps" else "auto",
                device_map=self.device
            )
        print("Model loaded successfully!")

    def generate_response(self, prompt, system_prompt="You are a helpful chatbot assistant.", history=None, max_new_tokens=512):
        """
        Generates a response from the LLM given a prompt and history.
        history: list of dicts, e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        if history:
            messages.extend(history)
            
        messages.append({"role": "user", "content": prompt})
        
        # Apply the model's chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate response
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Strip the input tokens from the generated output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

def main():
    print("=" * 60)
    print("Starting Open-Source Local Chatbot Interactive Session")
    print("=" * 60)
    
    chatbot = LocalChatbot()
    
    history = []
    print("\nChatbot is ready! Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            print("\033[92mAssistant:\033[0m ", end="", flush=True)
            response = chatbot.generate_response(user_input, history=history)
            print(response)
            
            # Keep history updated (limit history length to last 10 exchanges as per tasks requirements)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            if len(history) > 20: # 10 exchanges = 20 messages
                history = history[-20:]
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
