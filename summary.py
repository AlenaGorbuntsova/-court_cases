from transformers import AutoTokenizer
import re


tokenizer = AutoTokenizer.from_pretrained('marianna13/flan-t5-base-summarization')


def text_to_list(text):
    text = text.replace('\n\n', '<eos>').replace('\n', '<eos>')
    # string_ = re.sub(r'\.([\S])', r'<<<>>> \1', string_)
    text = re.sub(r'\B(?<! [A-Z]|Mr|No|Ex)([a-z])[.] ', r'\1.<eos>', text)
    text = text.replace('Ws.<eos>', 'Ws. ')
    # string_ = string_.replace('<<<>>> ', '.')

    list_text = text.split('<eos>')
    list_text = list(map(lambda x: x.replace('\n', '    ').strip(), list_text))

    return list_text


# why to chunkize: https://github.com/huggingface/transformers/issues/5204#issuecomment-1369389700
def chunkize(sentences_list, chunk_size = 512):
    chunks = [sentences_list[0]]

    for sentence in sentences_list[1:]:
        if token_len(chunks[-1] + sentence) <= chunk_size:
            chunks[-1] += ' ' + sentence
        else:
            chunks.append(sentence)

    return chunks


def token_len(str_sequence):
    global tokenizer

    tokens = tokenizer.encode(
        str_sequence,
        return_tensors='pt', 
        max_length=2 ** 20,
        truncation=True)

    return len(tokens[0])


def recursion_summarizing(case, model, chunk_size, min_length, max_length):
    if token_len(case) < chunk_size:
        return ''

    case_list = text_to_list(case)
    chunks = chunkize(case_list, chunk_size)

    force_summarization = False
    # for chunk in chunks:
    #     if token_len(chunk) < max_length:
    #         force_summarization = True


    output = model(chunks, min_length=min_length, max_length=max_length, early_stopping=True)

    output = [i['summary_text'] for i in output]
    output = ' '.join(output)
    output = output.strip().replace(' .', '.')

    string_output = '\n'.join(text_to_list(output))

    if token_len(output) > (chunk_size * 2) or force_summarization:
        string_output = recursion_summarizing(output, model, chunk_size, min_length, max_length)

    return string_output
