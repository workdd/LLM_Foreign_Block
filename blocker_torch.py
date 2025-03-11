import torch

foreign_lang_mask = None


def foreign_language_blocker(tokenizer, input_ids, logits):
    """중국어, 일본어, 러시아어 토큰을 차단하는 함수"""
    global foreign_lang_mask

    if foreign_lang_mask is None:
        # 어휘집의 모든 토큰 ID 생성
        token_ids = torch.arange(logits.size(-1))
        decoded_tokens = tokenizer.batch_decode(token_ids.unsqueeze(1), skip_special_tokens=True)

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
        foreign_lang_mask = torch.tensor([any(any(start <= ord(c) <= end for start, end in all_ranges) for c in token if c) for token in decoded_tokens]).to(
            logits.device
        )

    # 해당 토큰에 대한 로짓을 -inf로 설정
    logits[:, foreign_lang_mask] = -float("inf")
    return logits
