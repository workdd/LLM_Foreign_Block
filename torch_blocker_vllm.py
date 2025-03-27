import numpy as np
import torch
import time

foreign_lang_mask = None  # torch.bool 텐서로 캐시됨

def blocker(tokenizer, input_ids, logits):
    """중국어, 일본어, 러시아어 토큰을 마스킹하는 logit processor (Torch 기반 최적화)"""
    global foreign_lang_mask

    start_time = time.time()

    # logits: shape (batch_size, vocab_size)
    # -> numpy일 경우 강제 변환
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).to("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = logits.shape[-1]

    if foreign_lang_mask is None:
        token_ids = list(range(vocab_size))

        # 디코딩 (batch_decode 지원 시 활용)
        if hasattr(tokenizer, "batch_decode"):
            decoded_tokens = tokenizer.batch_decode([[i] for i in token_ids], skip_special_tokens=True)
        else:
            decoded_tokens = [tokenizer.decode([i]) for i in token_ids]

        # 마스킹할 문자 범위 정의
        def is_foreign(token):
            ranges = [
                (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAFF),  # Chinese
                (0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF),                      # Japanese
                (0x0400, 0x04FF), (0x0500, 0x052F),                                        # Russian
            ]
            return any(any(start <= ord(c) <= end for start, end in ranges) for c in token if c)

        mask_list = [is_foreign(token) for token in decoded_tokens]
        foreign_lang_mask = torch.tensor(mask_list, dtype=torch.bool, device=logits.device)

    # 마스킹 적용 (batch_size x vocab_size)
    logits[foreign_lang_mask] = float("-inf")

    print("logit 처리 시간:", time.time() - start_time)
    return logits
