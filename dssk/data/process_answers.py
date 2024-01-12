def postproc_tulu2(raw_answers: list[str]) -> list[str]:
    """Tülu-based models tend to used the prompt tags to end their answers,
    so cut everything starting with the first such tag in the answer:
    <|system|>, <|user|>, and <|assistant|>.
    It cannot handle situation where such tags are already truncated, due to
    being made of multiple tokens in the model.
    """
    answers = []
    for s in raw_answers:
        for tag in ["<|system|>", "<|user|>", "<|assistant|>"]:
            p = s.find(tag)
            if p != -1:
                s = s[:p]
        answers.append(s)
    return answers


KNOWN_ANSWER_PROCESSING = {
    "postproc_tulu2": postproc_tulu2,
}
