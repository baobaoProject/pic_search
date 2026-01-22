from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizerBase, T5Tokenizer

cache_dir = "D:\\000-Docker\\App\\Milvus-pic_search\\webserver\\models\\huggingface\\hub"


def clip_tokenizer_test():
    # 分词策略：基于Byte - Pair
    # Encoding（BPE）算法
    # 特殊标记：使用 < | startoftext | > 和 < | endoftext | > 作为特殊标记
    # 适用场景：多模态模型、图文匹配、图像分类等
    # 技术实现：纯Python实现（Tokenizer）和Rust加速实现（TokenizerFast）
    tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16",
                                              cache_dir=cache_dir)
    text = "我爱深度学习"
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)


def llama_tokenizer_test():
    # 分词策略：基于SentencePiece算法，支持多语言分词
    # 特殊标记：使用 < s > 作为开始标记， < / s > 作为结束标记
    # 适用场景：大语言模型、对话系统、复杂文本生成等
    # 技术实现：纯Python实现（Tokenizer）和Rust加速实现（TokenizerFast）
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir=cache_dir)
    text = "我爱深度学习"
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)


def t5_tokenizer_test():
    # 分词策略：基于SentencePiece算法，支持多语言分词
    # 特殊标记：使用 < s > 作为开始标记， < / s > 作为结束标记
    # 适用场景：多语言任务、文本摘要、机器翻译等
    # 技术实现：纯Python实现（Tokenizer）和Rust加速实现（TokenizerFast）
    tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir=cache_dir)
    text = "translate English to French: I love deep learning"
    inputs = tokenizer(text, return_tensors="pt")

    print(inputs)  # 输出：torch.Size([1, 11])


def gpt_tokenizer_test():
    # 分词策略：基于Byte-Pair Encoding（BPE）算法，将文本拆分为子词
    # 特殊标记：仅添加<|endoftext|>作为结束标记
    # 适用场景：单向语言模型、文本生成、代码生成等
    # 技术实现：纯Python实现（Tokenizer）和Rust加速实现（TokenizerFast）

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    text = "I love deep learning"
    inputs = tokenizer(text, return_tensors="pt")

    print(inputs["input_ids"])  # 输出：tensor([[40, 1842, 2784, 4083]])
    print(inputs["attention_mask"])  # 输出：tensor([[1, 1, 1, 1]])


def bert_tokenizer_test():
    # BERT系Tokenizer
    # 🎯 核心特征
    # 分词策略：基于WordPiece算法，将未登录词拆分为子词
    # 特殊标记：添加[CLS]（分类标记）和[SEP]（分隔标记）
    # 适用场景：双向语言模型、文本分类、命名实体识别等
    # 技术实现：纯Python实现（Tokenizer）和Rust加速实现（TokenizerFast）
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese",
                                                                       cache_dir=cache_dir)
    print(tokenizer.__class__)

    text = "我爱深度学习"
    encoded = tokenizer(text, return_tensors="pt")
    print(encoded)

    question = "什么是深度学习？"
    answer = "深度学习是一种基于神经网络的机器学习方法，它可以从大量数据中学习复杂的模式和特征。。"

    inputs = tokenizer(
        text=[question, answer],
        text_pair=[answer, question],
        max_length=50,
        padding="max_length",
        return_tensors="pt"
    )
    print(inputs)
