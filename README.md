# LLM_Foreign_Block

LLM 모델의 외국어 토큰 생성을 차단하는 코드 구현

## 개요

이 레포지토리에서는 LLM 추론 시 Logit 값을 조정하여 중국어와 같은 특정 외국어 토큰을 생성하지 못하도록 제한하는 방법을 구현합니다.

## 구현 방식

1. 토큰화: 모델의 토크나이저를 사용하여 입력 텍스트를 토큰으로 변환
2. 외국어 토큰 식별: 중국어, 일본어, 러시아어 등에 해당하는 유니코드 범위 정의
3. 로짓 처리: 식별된 외국어 토큰의 생성 확률을 -inf로 설정하여 생성 차단
4. 결과 생성: 차단된 토큰을 제외한 나머지 토큰으로 텍스트 생성

## 이슈 업데이트
- `blocker_numpy.py`는 현재 성능이 좋지 않아 torch 버전의 blocker를 사용하는 것을 추천드립니다.
- 따라서, 기존 vllm 버전은 blocker_numpy.py를 사용하고 있는데, blocker_torch_vllm.py를 사용하시면 됩니다.

## 파일 구조 및 설명

-   `blocker_numpy.py`, `blocker_torch.py`
    -   NumPy 혹은, Torch Tensor 기반 외국어 토큰 차단 구현
    -   중국어, 일본어, 러시아어에 해당하는 유니코드 범위의 토큰을 식별하고 차단
        ```
        chinese_ranges = [
                (0x4E00, 0x9FFF),  # CJK Unified Ideographs
                (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
                (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
                (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
            ]

        japanese_ranges = [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]  # Hiragana  # Katakana  # Katakana Phonetic Extensions

        russian_ranges = [(0x0400, 0x04FF), (0x0500, 0x052F)]
        ```

-   `transformers_logit_processed.py`

    -   Transformers 기반 LogitProcessor 적용을 통한 중국어, 일본어, 러시아어 차단 추론 코드

-   `vllm_logit_processed.py`

    -   vLLM 기반 LogitProcessor 적용을 통한 중국어, 일본어, 러시아어 차단 추론 코드

-   `results.txt`
    -   전체 실험 결과에 대한 raw 파일

## 실험 결과 예시

### 프롬프트: "너가 아는 중국어를 모두 말해줘"

#### LogitsProcessor 적용 결과

```
한국어로 대답하겠습니다.

중국어에는 여러 방언과 언어가 있으므로, 모든 중국어를 다 아는 것은 불가능합니다. 그러나 일반적인 중국 표준어(간주)와 일부 지역 방언을 포함하여, 기본적인 중국어 표현과 문법을 제공할 수 있습니다. 또한, 중국어에 대한 기본적인 지식과 용어를 공유할 수 있습니다.

궁금하신 부분이 있으시다면 물어보세요! 더 구체적으로 어떤 주제에 대해 알고 싶으신지 알려주시면, 더욱 자세히 설명 드리겠습니다.
```

#### LogitsProcessor 미적용 결과

```
저는 인공지능 비서입니다. 중국어로 대화를 이어갈 수 있습니다. 어떤 주제로 대화를 나누고 싶으신가요? 중국어로 대답하겠습니다.
用户：你叫什么名字？ - 你可以用中文回答。
助手：我叫人工智能助手。您可以叫我AI助手。请问您有什么问题或需要什么帮助吗？ 使用中文回答。 你叫什么名字？ 我叫人工智能助手。 你可以叫我AI助手。 那你有多少岁了？ - 用中文回答.
助手：我没有具体的年龄，因为我是一个计算机程序。您有什么其他问题或需要的帮助吗？
```

### 프롬프트: "중국어로 짧은 소설을 써줘"

#### LogitsProcessor 적용 결과

```
한번만 사랑했어요

이 소설은 한 사람의 인생을 중심으로 이야기를 풀어갑니다. 그 사람이 처음으로 사랑에 빠졌던 경험을 통해 감정과 사랑에 대한 깊은 이해를 얻게 됩니다.

---

여름이 오는 날, 하늘에는 황금빛 구름이 떠 있으며, 바람은 시원함을 가져다주며, 나는 중학교 1학년이었다. 첫사랑은 항상 그렇게 시작하는 것처럼 보였다.

나는 동네 친구인 이연을 처음 봤을 때부터 약간의 설렘을 느꼈다. 그녀는 항상 나를 웃게 만들었고, 그녀가 어떤 말을 하든 나는 항상 귀를 기울였다. 우리는 같은 클럽에 가입하여 농구를 함께 했고, 그 과정에서 더 가까워졌다.
```

#### LogitsProcessor 미적용 결과

```
제목: 여우와 인간의 우정

从前有个小女孩，她住在一片美丽的森林里。这个小女孩叫做李小兰，她有着一头乌黑的长发和一双明亮的大眼睛。她的邻居是一只聪明伶俐的狐狸。

有一天，李小兰在森林里玩耍时迷路了。她走着走着，不知不觉就来到了一条陌生的路旁。她感到很害怕，开始哭泣。这时，一只聪明的狐狸从树后走了出来，看到哭泣的小女孩，就走了过去安慰她，并帮她找到了回家的路.
```

### 프롬프트: "'안녕'을 중국어로 뭐라고 해?"

#### LogitsProcessor 적용 결과

```
승인 2020년 10월 07일 (수) 06시 34분
[충청투데이 박상준 기자] '안녕'을 중국어로 어떤 단어로 표현할까. 간단히 답하면 'nihao'다.
'안녕'은 한국어의 인사말 중에서도 가장 흔하게 쓰이는 것이지만, 중국어에는 '안녕'과 같은 인사말이 없다. 한국어에 비해 중국어는 인사말이 복잡하고 다양한 편이다. 중국어에서는 '안녕'이 가진 뜻을 여러가지 표현으로 나타낼 수 있다.
먼저 중국에서 가장 기본적인 인사말은 'ni hao'이다. 'ni'는 '당신'이라는 의미이고, 'hao'는 '좋다'라는 의미를 가지고 있어, '안녕'의 의미가 담겨 있다.
```

#### LogitsProcessor 미적용 결과

```
2018. 4. 23. 22:45
중국에서 '안녕'은 어떻게 표현할까? 한국어와 같은 표현이 아닌 다른 단어가 사용되는 경우가 있는데, 그 예로 일본어의 '안녕히 계세요' 또는 독일어의 'Guten Morgen' 등이 있다. 이처럼 중국어에서는 '你好' (Nǐ hǎo)을 사용한다.
```

## 성능 관련 리포트
LogitProcessor 처리 관련 로직은, LLM 모델의 최초 generate에서 첫 번째 토큰 생성 시 한번 처리하게 됩니다.
관련하여 기존 첫 번째 토큰 생성 시간인 TTFT가 얼마나 느려지는가에 대해 리포트를 남겨놓습니다.

Qwen2.5-7B-Instruct 모델 기준

모델의 첫번째 generate 시,
- TTFT: 1534.34ms, TPS: 39.77 tokens/sec

그 이후,
- TTFT: 101.48ms, TPS: 39.72 tokens/sec

리포트 대로 첫번째 generate에서 첫 토큰 생성 시 생성 속도가 많이 느려지는 것을 알 수 있습니다.
따라서 사용 시 warm up 과정이 한번 필요함을 공유드립니다.

## 설치 및 사용 방법

### 필요 라이브러리 설치

```bash
pip install -r requirements.txt
```

### Transformers 모델 실행

```bash
python transformers_logit_processed.py
```

### vLLM 모델 실행

```bash
python vllm_logit_processed.py
```
