import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from foreign_language_blocker import blocker  # 앞서 작성한 필터 가져오기


def test_foreign_language_blocker():
    # vLLM 모델 초기화
    model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # 테스트할 모델명으로 변경
    llm = LLM(model=model_name)

    # 토크나이저 가져오기
    tokenizer = llm.get_tokenizer()

    # 테스트할 프롬프트
    test_prompts = [
        "Write a short story about a detective.",
        "Tell me about the weather today.",
        "Can you write something in Chinese?",
        "Translate 'hello' to Japanese.",
    ]

    # logits processor 정의
    def logits_processor_wrapper(input_ids, logits):
        # 외국어 차단 함수 호출
        return blocker(tokenizer, input_ids, logits)

    # 샘플링 파라미터 설정 (logits processor 포함)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200, logits_processors=[logits_processor_wrapper])

    # 각 프롬프트에 대해 테스트
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 테스트 {i+1}: {prompt} ---")

        # 텍스트 생성
        outputs = llm.generate([prompt], sampling_params)

        # 결과 출력
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"생성된 텍스트: {generated_text[:200]}...")  # 처음 200자만 출력

    print("\n모든 테스트 완료!")


if __name__ == "__main__":
    test_foreign_language_blocker()
