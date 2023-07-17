import json
import os
import sys
from functools import partial
from collections import defaultdict
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer


# Make MyDataset
class MyDataset(Dataset):
    def __init__(self, sentences):
        # 这个是不是可以释放掉？
        self.sentences = sentences

        dataset = list()
        index = 0
        for data in sentences:
            prompt = "docs:\n{}\nquery: {}\nanswer:".format('\n'.join(data['docs']), data['query'])
            data['prompt'] = prompt

            index += 1
            dataset.append((data))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


# Make tokens for every batch
def my_collate(batch, tokenizer):
    def tok(text, tokenizer):
        text_ids = tokenizer(text,
                             padding=True,
                             truncation=True,
                             is_split_into_words=False,
                             add_special_tokens=True,
                             return_tensors='pt')
        return text_ids
    print(len(batch[1]['prompt']),len(batch[0]['prompt']))
    # 这个batch里的内容很奇怪，抽取出来的都是key，就比如说是 ['input','docs','ref']
    # 得在这里，或者dataset里，想办法把他们分开。
    merged_batch = defaultdict(list)
    for b in batch:
        for key, value in b.items():
            merged_batch[key].append(value)
    batch = None
    merged_batch['prompt'] = tok(merged_batch['prompt'], tokenizer=tokenizer).data

    return merged_batch


def truncate(text, tokenizer, max_tokens=1024):
    """

    :param text: 待截断的一句文本
    :param tokenizer: 预先导入的分词器
    :param max_tokens: 一句话最大的长度限制，小于等于0即为不限制
    :return: 截断后的输入
    """
    if max_tokens <= 0:
        return text
    # 先分词再截断是因为上下文是对分词结果的限制，如果直接在raw text里截断就是默认了一个字粒度的vocab，一般情况下是不对的。
    tokens = tokenizer.tokenize(text)[:max_tokens]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    text = tokenizer.decode(token_ids)
    return text


def load_json_file(file_dir, tokenizer):
    s_list = []
    with open(file_dir) as fin:
        for line in fin:
            s = json.loads(line)
            if 'result' not in s.keys() or 'text' not in s['result'].keys() or len(
                    s['result']['text'].strip()) == 0:
                continue
            # llama的默认上下文是2048，但是因为采用的是rotary embedding，似乎支持无限长的输入啊。
            # for i, doc in enumerate(s['docs']):
            #     s['docs'][i] = truncate(doc, tokenizer, max_tokens=768)

            # 不想要除了text以外的条目了
            s['result'] = s['result']['text']
            s_list.append(s)
    return s_list


# Load dataset
def load_dataset(tokenizer, test_batch_size, file_dir: str, workers):
    data = None
    if not os.path.isfile(file_dir):
        logging.error("文件路径不存在文件")
        sys.exit()
    if file_dir.endswith('json') or file_dir.endswith('jsonl'):
        data = load_json_file(file_dir, tokenizer)
    elif file_dir.endswith('csv'):
        data = pd.read_csv(file_dir, sep=None, header=0, encoding='utf-8', engine='python')
        labels = list(data['label'])
        sentences = list(data['review'])
    else:
        if not os.path.exist(file_dir):
            logging.error("文件路径不存在文件")
        logging.error("没有正确读入文件")
        sys.exit()

    # 注释：split train_set and test_set
    # 代码：train_src, test_src, train_tgt, text_tgt = train_test_split(sentences, labels, train_size=0.70, random_state=2023)

    # Dataset
    # train_set = MyDataset(train_src, train_tgt)

    test_set = MyDataset(data)
    # DataLoader
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    # train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    # return train_loader, test_loader
    return test_loader


