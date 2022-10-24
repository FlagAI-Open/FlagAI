# Examples in datasets

## SuperGLUE: BoolQ
| 键值                          | 含义  |
|------------------------------|----------|
| passage                      | 背景信息  |
| question                     |  根据背景信息提出的是/否问题  |
| label                        | 取值范围：true/false, 分别代表答案为是/否 |
示例
```json
{"question": "is barq's root beer a pepsi product", 
 "passage": "Barq's -- Barq's is an American soft drink. Its brand of root beer is notable for having caffeine. Barq's, created by Edward Barq and bottled since the turn of the 20th century, is owned by the Barq family but bottled by the Coca-Cola Company. It was known as Barq's Famous Olde Tyme Root Beer until 2012.", 
 "idx": 5, "label": false}
```

## SuperGLUE: CB
| 键值                          | 含义  |
|------------------------------|----------|
| premise                      | 前提文本  |
| hypothesis                     |  假设文本  |
| label                        |取值范围：entailment/contradiction/neutral 分别代表前提与假设的关系为1.前提能推导出假设 2.前提与假设矛盾 3. 没有足够信息得到两者关联性<br /> |
示例
```json
{"premise": "Mary is a high school student.",
 "hypothesis": "Mary is a student",
 "label": "entailment", "idx": 10}
```

## SuperGLUE: Copa
| 键值                          | 含义  |
|------------------------------|----------|
| premise                      | 前提文本  |
| choice1                      |  选项1  |
| choice2                      |选项2|
| question                      | 取值范围： cause/effect, 分别代表问题是1.两个选项里哪个是前提的原因2.两个选项里哪个是前提的结果  |
| label                     | 取值范围：0/1, 分别代表选项1为正确答案和选项2为正确答案  |
示例
```json
{"premise": "My eyes became red and puffy.",
 "choice1": "I was sobbing.", "choice2": "I was laughing.",
  "question": "cause", "label": 0, "idx": 7}
```

## SuperGLUE: MultiRC
| 键值                          | 含义  |
|------------------------------|----------|
| text                      | 背景信息  |
| questions         |  包含了一系列针对背景信息提出的问题，其中每个问题下有五个回答，而每个回答后面有一个标签，标签为0代表此回答正确，标签为1代表此回答错误  |
示例
```json
{"idx": 4, "version": 1.1, 
 "passage": {"text": "...The two companies numbered about 225 men, and were commanded by General John E. Ross, a veteran Indian fighter... ", 
              "questions": [{"question": "When the narrator arrived at the headquarters, approximately how many men were present?", 
                              "answers": [{"text": "225 men", "idx": 178, "label": 1}, 
                                          {"text": "225", "idx": 179, "label": 1}, 
                                          {"text": "525 men", "idx": 180, "label": 0}, 
                                          {"text": "235", "idx": 181, "label": 0}, 
                                          {"text": "255 men", "idx": 182, "label": 0}], "idx": 41}]}}
```


## SuperGLUE: RTE
| 键值                          | 含义  |
|------------------------------|----------|
| premise                      | 前提文本  |
| hypothesis                     |  假设文本  |
| label                        |取值范围：entailment/not entailment 分别代表前提与假设的关系为1.前提能推导出假设 2.前提不能推导出假设|
示例
```json
{"premise": "Security forces were on high alert after an election campaign in which more than 1,000 people, including seven election candidates, have been killed.", 
"hypothesis": "Security forces were on high alert after a campaign marred by violence.", 
"label": "entailment", "idx": 4}
```


## SuperGLUE: Wic
| 键值                          | 含义  |
|------------------------------|----------|
| sentence1                      | 背景文本1  |
| sentence2                      |  背景文本2  |
| word                      | 一个在sentence1和sentence2中都会出现的单词 |
| label                     | 取值范围：true/false, 分别代表选项1.word在sentence1和sentence2中有着相同的含义 2.在两段话里word的含义不同  |
示例
```json
{"word": "class", "sentence1": "An emerging professional class.", 
"sentence2": "Apologizing for losing your temper, even though you were badly provoked, showed real class.",
 "idx": 0, "label": false, "start1": 25, "start2": 85, "end1": 30, "end2": 90, "version": 1.1}
```

## SuperGLUE: WSC
| 键值                          | 含义                                                                         |
|------------------------------|----------------------------------------------------------------------------|
| text                        | 背景文本                                                                       |
| span2_text                      | 背景文本中出现过的某个代词                                                              |
| span1_text                      | 背景文本中出现过的某个片段                                                              |
| label                     | 取值范围：true/false, 分别代表了1.span1_text这个代词指向了span2_text中的内容 2.没有指向到span2_text。 |
示例
```json
{"word": "class", "sentence1": "An emerging professional class.", 
"sentence2": "Apologizing for losing your temper, even though you were badly provoked, showed real class.",
 "idx": 0, "label": false, "start1": 25, "start2": 85, "end1": 30, "end2": 90, "version": 1.1}
```

## CLUE: AFQMC
| 键值                          | 含义  |
|------------------------------|----------|
| sentence1                      | 背景文本1  |
| sentence2                      |  背景文本2  |
| label                     | 取值范围：0/1, 分别代表选项1.两句话的含义不同 2.两句话的含义相似  |
示例
```json
{"sentence1": "双十一花呗提额在哪", "sentence2": "里可以提花呗额度", "label": "0"}
```

## CLUE: TNEWS
| 键值                          | 含义  |
|------------------------------|----------|
| sentence                      | 背景文本  |
| label                      |  背景文本的标签，包含十五种类别   |
| label_des                     |  标签对应的真正含义  |
示例
```json
{"label": "102", "label_des": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}
```


## CLUE: CMRC2018
| 键值                          | 含义  |
|------------------------------|----------|
| context                      | 背景信息  |
| question                      |  根据背景信息提出来的问题，一段背景信息可以对应多个问题   |
| answers                    |  对于某个问题，answers包含了三个人工给出来的回答  |
示例
```
{'paragraphs': 
    [{'id': 'TRAIN_186', 
    'context': '工商协进会报告...所谓的“傻钱”策略，其实就是买入并持有美国股票这样的普通组合。...',
    'qas':{'question': '消费者信心指数由什么机构发布？', 'id': 'TRAIN_186_QUERY_4', 
    'answers': [{'text': "工商协进会", 'answer_start': 759},
                 {'text': "工商协进会", 'answer_start': 759},
                 {'text': "工商协进会", 'answer_start': 759}]}]}], 
    'id': 'TRAIN_186', 'title': '范廷颂'}
```