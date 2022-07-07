# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# print(tokenizer.tokens)
import sys
sys.path.append("..")
# import os
# print(os.getcwd())
# from examples.uni_tokenizer.base_tokenizer import BaseTokenizer
# tokenizer = BaseTokenizer.from_pretrained('GLM-large-en')

# from flagai.data.tokenizer import GLMLargeChTokenizer, GLMTokenizer
#
# model_name = 'GLM-large-ch'
# tokenizer = GLMTokenizer.from_pretrained(model_name)
# print(tokenizer.DecodeIds([43358]),123)
from flagai.data.tokenizer import GLMLargeChTokenizer
tokenizer = GLMLargeChTokenizer()
print(tokenizer.DecodeIds([43358]),321)
# from flagai.data.tokenizer.glm_large_en.wordpiece import GLMLargeEnTokenizer
# from flagai.data.tokenizer.glm_10b_en.glm_10b_en_tokenizer import GLM10bENTokenizer
#
# tokenizer = GLMLargeEnTokenizer.from_pretrained("GLM-large-en")
# # tokenizer = GLM10bENTokenizer.from_pretrained("gpt2")
# print(tokenizer.vocab['[CLS]'])
# print(tokenizer.vocab['[UNK]'])
# print(tokenizer.vocab['[SEP]'])
# print(tokenizer.vocab['<|startofpiece|>'])
# print(tokenizer.vocab['<|endofpiece|>'])
# print(tokenizer.vocab_size())
# print(list(dict(tokenizer.encoder).items())[-2:])

# class Animal(object):
#     @classmethod
#     def move(cls):
#         # return cls.jump(cls,8,12)
#         return cls(8,12)
#     def __init__(self, name=None, age=None):
#         print(name, age)
#         # super(Animal, self).__init__()
#         self.name = name
#         self.age = age
#         print('parent')
#
#     # def jump(self,  name, age):
#     #     print("jump")
#
# class cat(Animal):
#     def __init__(self, age, piece, **kwargs):
#         super().__init__(**kwargs)
#         self.age = age+age
#         self.piece = 8
#         print("is it?")
#
#
#     def jump(self,  name, age):
#         self.age = 13
#         print(self.age)
#
# a = cat.move()
# a.jump(2,4)
# a = cat(age=9,piece=7).move()
# print(a.piece)

# class FooParent(object):  #⽗类
#     def __init__(self):
#         self.parent = 'I\'m the parent.'
#         print('Parent')
#     def bar(self,message):
#         print(message, 'from Parent')
# class FooChild(FooParent):  #⼦类
#     def __init__(self):
#         # FooParent.__init__(self)  #调⽤未绑定的超类构造⽅法【必须显式调⽤⽗类的构造⽅法，否则不会执⾏⽗类构造⽅法，这个跟Java
#         print('Child')
#     def bar(self,message):
#         FooParent.bar(self,message)
#         print('Child bar function.')
#         print(self.parent)
# if __name__=='__main__':
#     fooChild = FooChild()
#     # fooChild.bar('HelloWorld')
