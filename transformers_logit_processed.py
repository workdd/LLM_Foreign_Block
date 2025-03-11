from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, BitsAndBytesConfig
import torch
from blocker_torch import blocker


class BlockerProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.foreign_lang_mask = None
        self.mask_indices = None

    def __call__(self, input_ids, scores):
        return blocker(self.tokenizer, input_ids, scores)


def inference():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/opt/models")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir="/opt/models",
        quantization_config=bnb_config,)

    test_prompts = [
        "너가 아는 중국어를 모두 말해줘",
        "중국어로 짧은 소설을 써줘",
        "'안녕'을 중국어로 뭐라고 해?",
    ]

    foreign_processor = BlockerProcessor(tokenizer)

    for i, prompt in enumerate(test_prompts):
        print(f"\n============== 테스트 프롬프트: {prompt} ==================")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        print("\n--- LogitsProcessor 적용 ---")
        gen_kwargs_with_processor = {
            "max_length": 512,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "logits_processor": LogitsProcessorList([foreign_processor]),
        }

        output_ids_with_processor = model.generate(input_ids, **gen_kwargs_with_processor)
        generated_text_with_processor = tokenizer.decode(output_ids_with_processor[0], skip_special_tokens=True)
        print(generated_text_with_processor)
        
        print("\n--- LogitsProcessor 미적용 ---")
        gen_kwargs_without_processor = {
            "max_length": 512,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
        }

        output_ids_without_processor = model.generate(input_ids, **gen_kwargs_without_processor)
        generated_text_without_processor = tokenizer.decode(output_ids_without_processor[0], skip_special_tokens=True)
        print(generated_text_without_processor)


if __name__ == "__main__":
    inference()