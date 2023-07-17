from collections import defaultdict

from transformers import LlamaForCausalLM, LlamaTokenizer

from dataset import load_dataset


def get_tokenizer_and_model(llama_path):  # 返回分词器和模型
    tokenizer = LlamaTokenizer.from_pretrained(llama_path, padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = LlamaForCausalLM.from_pretrained(llama_path)
    model.half()
    model.cuda()

    # model.model.padding_idx=tokenizer.convert_tokens_to_ids('[PAD]')

    return tokenizer, model


# print(tokenizer(a, padding=True,
#                 truncation=True,
#                 is_split_into_words=False,
#                 add_special_tokens=True,
#                 return_tensors='pt'))
def get_ngrams(tokens, n):
    ngram_list = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i + n])
        ngram_list.append(ngram)
    return ngram_list


def get_batch_ngrams(batch_tokens, n):
    """

    Args:
        batch_tokens: 一个批次的话
        n:
        pad_idx: 之所以需要pad的索引，是因为对batch数据取ngram很可能会取到这个玩意。

    Returns:

    """
    ngram_list = []
    for tokens in batch_tokens:
        tokens = get_ngrams(tokens, n)
        ngram_list.append(tokens)
    return ngram_list


def prepare_batch_ngrams(batch, n, tokenizer, max_n=0):
    """

    :param batch:
    :param n:   匹配长度下界
    :param tokenizer:
    :param max_n:   匹配长度上界
    :return: 返回一个dict，里面有tokenize之后的docs和result，也有n-maxn长度的对应ngram列表，还有docs的id？
    """
    # 生成[n,max_n]长度的所有gram。最后对结果做逆序，也就是更长的ngram排在return的参数里的更前面
    if max_n < n:
        max_n = n
    docs = batch['docs']
    if 'result' in batch.keys():
        gtext = batch['result']
        gtokens = list(map(lambda x: tokenizer.tokenize(x), gtext))
        g_ngrams_list = []
        for gtoken in gtokens:
            gtoken_ngram=[]
            for l in range(n, max_n + 1):
                # 本身就是一个list，内容是['ngram1',..'ngramM']
                g_ngrams = get_ngrams(gtoken, l)
                gtoken_ngram.append([l,g_ngrams])
            g_ngrams_list.append(gtoken_ngram)
        for g in g_ngrams_list:
            g.reverse()
        # g_ngrams_list.reverse()
    else:
        gtokens = None
        g_ngrams_list = None

    doc_list = [[tokenizer.tokenize(y) for y in x] for x in batch['docs']]
    doc_token_id_list = [[tokenizer.convert_tokens_to_ids(y) for y in x] for x in doc_list]

    doc_ngrams_list = [[] for _ in range(len(doc_list))]

    # doc_tokens_list是一个示例的10个doc分词列表，维度是一个二维数组，第一维是第几个doc，第二维是第几个词
    for i, doc_tokens_list in enumerate(doc_list):
        # doc_ngrams里key是ngram，值是一个元组(i,j)，内容是第i个doc的第j个ngram

        for l in range(n, max_n + 1):
            doc_ngrams = defaultdict(list)
            # 是一个batch的，l长度的gram list，里面应该是
            ngram_list = get_batch_ngrams(doc_tokens_list, l)
            for j, doc_ngram in enumerate(ngram_list):
                for k, ngram in enumerate(doc_ngram):
                    # i是第batch的第几句话，j是第几个doc，k是doc的第几个ngram
                    doc_ngrams[ngram].append((j, k))

            doc_ngrams_list[i].append([l, doc_ngrams])
    # 小心这个reverse……因为第0维可能不太一样
    for d in doc_ngrams_list:
        d.reverse()
    # doc_ngrams_list.reverse()

    # gtokens，，因为一条示例有很多doc，所以才会变成[[]] ； doc_token_id_list是doc_list独热的结果
    # g_ngrams_list，doc_ngrams_list是按n大小降序排的ngram数组
    return {"target_ngrams": g_ngrams_list,  # 维度是(bsz,maxN-triggerN+1,2,) 存储的元素是 最后的2存储的就是 l和lgram
            "doc_list": doc_list,  # 一个batch的docs分词后的结果，维度是(bsz , 10)
            "doc_ngrams": doc_ngrams_list,  # (bsz,maxN-triggerN+1,2) 存储的元素是一批次句子的对应的ngram
            "target_tokens": gtokens, # 类似 doc_list (bsz,10,seqlen)
            "doc_token_id_list": doc_token_id_list} # doc list 上转成id的列表(bsz,10，seq_len)


# for t in test_dataloader:
#     # a=tokenizer.batch_decode(t['prompt']['input_ids'])
#     b = prepare_batch_ngrams(t, 2, tokenizer, max_n=5)
#     print(a)
#     break
def run_batch_time_test(dataloder, decoding_fn, model, tokenizer, trigger_N, block_K, append_docs=True,
                        forced_decoding=False):
    '''

    Args:
        s_list:
        decoding_fn:
        model:
        tokenizer:
        trigger_N:
        block_K:
        append_docs:
        forced_decoding:

    Returns:

    '''
    for batch in dataloder:
        # gtokens，doc_list里是分词了的结果，因为一条示例有很多doc，所以才会变成[[]] ； doc_token_id_list是doc_list独热的结果
        # g_ngrams_list，doc_ngrams_list是按n大小降序排的ngram数组
        ngrams_cache = prepare_batch_ngrams(batch, trigger_N, tokenizer, max_n=5)
        if 'result' in batch.keys() and 'text' in batch['result']:
            # 看了下数据，text开头总是有一个空格，所以[:, 1:]是正常的。不过这个为啥不在prepare_ngrams里头生成呀
            gen_texts_ids = tokenizer(batch['result']['text'], return_tensors="pt").input_ids[:, 1:]
        else:
            gen_texts_ids = None
        batch["ngrams_cache"] = ngrams_cache
        ngrams_cache = None
        batch["gen_texts_ids"] = gen_texts_ids
        query = batch['query']
        if append_docs:
            # docs = '\n'.join(s['docs'])
            prompt = batch['prompt']
        else:
            prompt = tokenizer(query, return_tensors="pt")
        # inputs = tokenizer(prompt, return_tensors="pt")
        # s["inputs"] = inputs
    # 上面看起来其实有一点自欺欺人  ，因为他尝试把一些本不存在输入里的东西封进去，我猜这有不小的时间开销
    # TODO test 把start time提到开头，测试一下
    # 下面如果想和上面融合的话，还比较复杂，就是
        acc_time = 0  # 压根没用上！
        total_length = 0
        total_start_time = time.time()
        # 这里有一个很大的问题，他是只支持逐个句子解码的？ 如果我想封成一个batch，还需要一些额外的东西，比如dataset和collate。
        for s in tqdm(s_list):
            start_time = time.time()
            inputs = s["inputs"]
            ngrams_cache = s["ngrams_cache"]
            gen_texts_ids = s["gen_texts_ids"]

            generate_ids = decoding_fn(model, tokenizer, inputs.input_ids, gen_texts_ids, trigger_N=trigger_N,
                                       block_K=block_K, forced_decoding=forced_decoding, ngrams_cache=ngrams_cache)
            total_length = generate_ids.shape[-1] + total_length
            end_time = time.time()
            s_time = end_time - start_time
            acc_time = s_time + acc_time
            generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                0]
            s["output"] = generated
            print(generated)
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        return total_time


# llama_path = '/root/autodl-tmp/chinese-alpaca-plus-7b-merged'
llama_path = r'C:\Users\lenovo\Desktop\pythonProject\LMOps-main\llma\src\chinese-alpaca-plus-7b-merged'
tokenizer = LlamaTokenizer.from_pretrained(llama_path, padding_side='left')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

test_dataloader = load_dataset(tokenizer, test_batch_size=2, file_dir='data/rag.jsonl', workers=0)

run_batch_time_test(dataloder=test_dataloader, decoding_fn=None, model=None, tokenizer=tokenizer, trigger_N=1,
                    block_K=3, append_docs=True,
                    forced_decoding=False)
