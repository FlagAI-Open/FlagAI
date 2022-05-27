# Examples in datasets

## SuperGLUE: BoolQ
| Key      | Meaning                                 |
|----------|-----------------------------------------|
| passage  | Background information                  |
| question | true/false question according to passage |
| label    | Range：true/false                        |
Example
```json
{"question": "is barq's root beer a pepsi product", 
 "passage": "Barq's -- Barq's is an American soft drink. Its brand of root beer is notable for having caffeine. Barq's, created by Edward Barq and bottled since the turn of the 20th century, is owned by the Barq family but bottled by the Coca-Cola Company. It was known as Barq's Famous Olde Tyme Root Beer until 2012.", 
 "idx": 5, "label": false}
```

## SuperGLUE: CB
| Key        | Meaning                                |
|------------|----------------------------------------|
| premise    | Premise text                           |
| hypothesis | Hypothesis text                        |
| label      | Range：entailment/contradiction/neutral |
Example:
```json
{"premise": "Mary is a high school student.",
 "hypothesis": "Mary is a student",
 "label": "entailment", "idx": 10}
```

## SuperGLUE: Copa
| Key      | Meaning                                                                                                             |
|----------|---------------------------------------------------------------------------------------------------------------------|
| premise  | Background information                                                                                              |
| choice1  | The first choice                                                                                                    |
| choice2  | The second choice                                                                                                   |
| question | Range： cause/effect, which is asking 1.which choice is the cause of premise 2.which choice is the effect of premise |
| label    | Range：0/1, which represents the answer is choice1 and choice2 , respectively                                        |
示例
```json
{"premise": "My eyes became red and puffy.",
 "choice1": "I was sobbing.", "choice2": "I was laughing.",
  "question": "cause", "label": 0, "idx": 7}
```

## SuperGLUE: MultiRC
| Key       | Meaning                                                                                                                                                                  |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| text      | Background information                                                                                                                                                   |
| questions | Consists of several questions for text. There are 5 answers under each question, and there is a label for each answer, label is 0 if the answer is correct, otherwise 1. |
Example:
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
| Key        | Meaning                           |
|------------|-----------------------------------|
| premise    | Premise text                      |
| hypothesis | Hypothesis text                   |
| label      | Range：entailment/not entailment   |
Example
```json
{"premise": "Security forces were on high alert after an election campaign in which more than 1,000 people, including seven election candidates, have been killed.", 
"hypothesis": "Security forces were on high alert after a campaign marred by violence.", 
"label": "entailment", "idx": 4}
```


## SuperGLUE: Wic
| Key       | Meaning                                                                                                                                |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------|
| sentence1 | sentence1                                                                                                                              |
| sentence2 | sentence2                                                                                                                              |
| word      | A word that appears in both sentence1 and sentence2.                                                                                   |
| label     | Range：true/false, which represents 1. This word has the same meaning in sentence1 and sentence2 2.Different meanings in two sentences. |
Example
```json
{"word": "class", "sentence1": "An emerging professional class.", 
"sentence2": "Apologizing for losing your temper, even though you were badly provoked, showed real class.",
 "idx": 0, "label": false, "start1": 25, "start2": 85, "end1": 30, "end2": 90, "version": 1.1}
```

## SuperGLUE: WSC
| Key        | Meaning                                                                           |
|------------|-----------------------------------------------------------------------------------|
| text       | Background information                                                            |
| span2_text | A pronoun in text                                                                 |
| span1_text | A sentence piece in text                                                          |
| label      | Range：true/false, which represents 1.span1_text refers to span2_text 2. elsewise. |
Example
```json
{"word": "class", "sentence1": "An emerging professional class.", 
"sentence2": "Apologizing for losing your temper, even though you were badly provoked, showed real class.",
 "idx": 0, "label": false, "start1": 25, "start2": 85, "end1": 30, "end2": 90, "version": 1.1}
```

## CLUE: AFQMC
| Key       | Meaning                                                                                           |
|-----------|---------------------------------------------------------------------------------------------------|
| sentence1 | sentence1                                                                                         |
| sentence2 | sentence1                                                                                         |
| label     | Range：0/1, Which represents 1.The meanings of two sentences are different 2.Meanings are similar. |
Example
```json
{"sentence1": "双十一花呗提额在哪", "sentence2": "里可以提花呗额度", "label": "0"}
```

## CLUE: TNEWS
| Key       | Meaning                         |
|-----------|---------------------------------|
| sentence  | Background information          |
| label     | labels which have 15 categories |
| label_des | What the label refers to        |
Example
```json
{"label": "102", "label_des": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}
```


## CLUE: CMRC2018
| Key      | Meaning                                                               |
|----------|-----------------------------------------------------------------------|
| context  | Background information                                                |
| question | Question for context, there could be multiple questions for one text. |
| answers  | for a question, there are 3 answers provided by humans                |
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