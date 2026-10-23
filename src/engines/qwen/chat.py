from data import speaker_names


def clean_reply(text, user_name, her_name):
    labels = (f"{her_name}:", f"{user_name}:")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(f"{user_name}:"):
            break
        for label in labels:
            if line.startswith(label):
                line = line[len(label):].strip()
                break
        line = line.lstrip(":").strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _history_rows(history, user_name, her_name):
    msgs = []
    for line in history:
        if line.startswith(f"{user_name}: "):
            msgs.append({"role": "user",
                         "content": line[len(f"{user_name}: "):]})
        elif line.startswith(f"{her_name}: "):
            msgs.append({"role": "assistant",
                         "content": line[len(f"{her_name}: "):]})
    return msgs


def build_messages(system, prompt, history):
    # demos stay in the system text: a chat template has no scene-break row,
    # so a demo in the message list reads as your own turn
    user_name, her_name = speaker_names()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(_history_rows(history, user_name, her_name))
    messages.append({"role": "user", "content": prompt})
    return messages


def generate(hf_tok, hf_model, messages, args):
    import engines.qwen.qwen_lm as qwen_lm
    return qwen_lm.reply(
        hf_tok, hf_model, messages,
        max_new_tokens=args.max_reply, temperature=args.temperature,
        top_p=args.top_p, repeat_penalty=args.repeat_penalty)
