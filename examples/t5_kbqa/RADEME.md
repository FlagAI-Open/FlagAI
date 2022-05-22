Knowledge Base Question Answering (KBQA) aims to answer natural language questions with
the help of an external knowledge base. The core idea is to find the link between the internal
knowledge behind questions and known triples of the knowledge base. Traditional KBQA task
pipelines contain several steps, including entity recognition, entity linking, answering selection,
etc. In this kind of pipeline methods, errors in any procedure will inevitably propagate to the
final prediction. To address this challenge, the Corpus Generation - Retrieve
Method (CGRM)   with Pre-training Language Model (PLM) for the KBQA task  aims to generate natural language QA pairs based on Knowledge Graph triples
and directly solve the QA by retrieving the synthetic dataset. The new method can extract more
information about the entities from PLM to improve accuracy and simplify the processes. 

The major procedure of CGRM is to generate QA pairs, in which, it needs to do two tasks: 
The first task is a [NER](finetune_t5_ner.py) downstream task. The second one is a downstram task for [generating natural language QA pairs based on Knowledge Graph triples](finetune_t5_triple2question.py).