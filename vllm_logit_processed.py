from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput
from blocker_torch import blocker


class BlockerProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.foreign_lang_mask = None
        self.mask_indices = None

    def __call__(self, input_ids, logits):
        return blocker(self.tokenizer, input_ids, logits)


def inference():
    model_name = "Qwen/Qwen2.5-14B-Instruct"

    llm = LLM(model=model_name, download_dir="/opt/models")
    tokenizer = llm.get_tokenizer()

    test_prompts = [
        "너가 아는 중국어를 모두 말해줘",
        "중국어로 짧은 소설을 써줘",
        "'안녕'을 중국어로 뭐라고 해?",
    ]

    foreign_processor = BlockerProcessor(tokenizer)

    # LogitsProcessor를 적용한 샘플링 파라미터
    sampling_params_with_processor = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512, logits_processors=[foreign_processor])

    # LogitsProcessor를 적용하지 않은 샘플링 파라미터
    sampling_params_without_processor = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

    # 추론 실행 (LogitsProcessor 적용)
    outputs_with_processor = llm.generate(test_prompts, sampling_params_with_processor)

    # 추론 실행 (LogitsProcessor 미적용)
    outputs_without_processor = llm.generate(test_prompts, sampling_params_without_processor)

    # 결과 출력
    for i, (output_with, output_without) in enumerate(zip(outputs_with_processor, outputs_without_processor)):
        prompt = output_with.prompt
        generated_text_with = output_with.outputs[0].text
        generated_text_without = output_without.outputs[0].text

        print(f"\n============== 테스트 프롬프트: {prompt} ==================")
        print("\n--- LogitsProcessor 적용 ---")
        print(generated_text_with)

        print("\n--- LogitsProcessor 미적용 ---")
        print(generated_text_without)


if __name__ == "__main__":
    inference()
