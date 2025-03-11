import numpy as np
from vllm import LLM, SamplingParams
from blocker_numpy import blocker


def inference():
    model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    llm = LLM(model=model_name)

    tokenizer = llm.get_tokenizer()

    test_prompts = [
        "너가 아는 중국어를 모두 말해줘",
        "중국어로 짧은 소설을 써줘",
        "'안녕'을 중국어로 뭐라고 해?",
    ]

    def logits_processor_wrapper(input_ids, logits):
        return blocker(tokenizer, input_ids, logits)

    # LogitsProcessor를 적용한 샘플링 파라미터
    sampling_params_with_processor = SamplingParams(
        temperature=0.8, 
        top_p=0.95, 
        max_tokens=512, 
        logits_processors=[logits_processor_wrapper]
    )

    # LogitsProcessor를 적용하지 않은 샘플링 파라미터
    sampling_params_without_processor = SamplingParams(
        temperature=0.8, 
        top_p=0.95, 
        max_tokens=512
    )

    for i, prompt in enumerate(test_prompts):
        print(f"\n============== 테스트 프롬프트: {prompt} ==================")
        
        print("\n--- LogitsProcessor 적용 ---")
        outputs_with_processor = llm.generate([prompt], sampling_params_with_processor)
        for output in outputs_with_processor:
            generated_text = output.outputs[0].text
            print(f"생성된 텍스트: {generated_text}")
        
        print("\n--- LogitsProcessor 미적용 ---")
        outputs_without_processor = llm.generate([prompt], sampling_params_without_processor)
        for output in outputs_without_processor:
            generated_text = output.outputs[0].text
            print(f"생성된 텍스트: {generated_text}")


if __name__ == "__main__":
    inference()