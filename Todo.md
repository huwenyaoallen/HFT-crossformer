# 2024.5.14
1. 测试不同输入长度以及不同seg_len的结果，选定一个seg_len用于后续的实验
2. 在数据load形式不变的情况下增加两个Baseline: 1)Transformer(同一个timestampembedding到一起）+maxpooling+MLP 2)直接将T*d flatten后过MLP
3. 拓展到多支股票
