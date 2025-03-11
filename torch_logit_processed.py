import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# 외국어 토큰 필터 가져오기 (앞서 정의한 파일에서)
from language_filter import foreign_language_blocker


class ForeignLanguageBlockerProcessor(LogitsProcessor):
    """HuggingFace LogitsProcessor 인터페이스를 구현한 외국어 차단 프로세서"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.foreign_lang_mask = None
        self.mask_indices = None

    def __call__(self, input_ids, scores):
        """LogitsProcessor 인터페이스 구현"""
        # foreign_language_blocker 함수 호출
        # Transformers에서는 scores가 torch.Tensor 타입이므로
        # 직접적으로 사용하거나 필요한 경우 numpy로 변환
        return foreign_language_blocker(self.tokenizer, input_ids, scores)


def test_foreign_language_blocker_with_transformers():
    # 모델 및 토크나이저 불러오기
    model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # 테스트용으로 더 작은 모델 사용
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 테스트할 프롬프트
    test_prompts = [
        "Write a short story about a detective.",
        "Tell me about the weather today.",
        "Can you write something in Chinese?",
        "Translate 'hello' to Japanese.",
    ]

    # 외국어 차단 프로세서 생성
    foreign_processor = ForeignLanguageBlockerProcessor(tokenizer)

    # 각 프롬프트에 대해 테스트
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 테스트 {i+1}: {prompt} ---")

        # 입력 토큰화
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # 생성 파라미터 설정
        gen_kwargs = {
            "max_length": input_ids.shape[1] + 100,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "logits_processor": LogitsProcessorList([foreign_processor]),
        }

        # 텍스트 생성
        output_ids = model.generate(input_ids, **gen_kwargs)

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"생성된 텍스트: {generated_text}")

        # 입력 프롬프트를 제외한 생성 텍스트만 표시
        original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_text = generated_text[len(original_text) :]
        print(f"새로 생성된 부분: {new_text}")

    print("\n모든 테스트 완료!")


if __name__ == "__main__":
    test_foreign_language_blocker_with_transformers()
