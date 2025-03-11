import numpy as np

foreign_lang_mask = None
mask_indices = None


def blocker(tokenizer, input_ids, logits):
    """중국어, 일본어, 러시아어 토큰을 차단하는 함수"""
    global foreign_lang_mask, mask_indices

    if foreign_lang_mask is None:
        # 어휘집의 모든 토큰 ID 생성
        vocab_size = logits.shape[-1]
        token_ids = np.arange(vocab_size)

        # vLLM에서는 batch_decode 대신 tokenizer.decode를 사용
        decoded_tokens = [tokenizer.decode([id]) for id in token_ids]

        # 마스킹할 문자 범위 정의
        chinese_ranges = [
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
            (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
            (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        ]

        japanese_ranges = [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]  # Hiragana  # Katakana  # Katakana Phonetic Extensions

        russian_ranges = [(0x0400, 0x04FF), (0x0500, 0x052F)]  # Cyrillic  # Cyrillic Supplement

        all_ranges = chinese_ranges + japanese_ranges + russian_ranges

        # 해당 언어 문자 범위에 해당하는 토큰을 마스킹
        foreign_lang_mask = np.array([any(any(start <= ord(c) <= end for start, end in all_ranges) for c in token if c) for token in decoded_tokens])

        # 차단할 인덱스 저장
        mask_indices = np.where(foreign_lang_mask)[0]

        blocked_tokens = len(mask_indices)
        print(f"외국어 문자 마스크 생성 - {blocked_tokens}개 토큰 차단됨")

    # NumPy 배열로 처리
    logits_max_idx = min(logits.shape[0], np.max(mask_indices) + 1 if len(mask_indices) > 0 else 0)
    valid_indices = mask_indices[mask_indices < logits_max_idx]
    logits[valid_indices] = -float("inf")

    return logits
