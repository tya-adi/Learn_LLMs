# Learning how tokenization is done in Large Language Models
# Byte Pair Encoding



import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')

text = "Hello World"

tokens = tokenizer.encode(text)

# print(tokens)
#
# print(tokenizer.decode([2159]))


# leading spaces are considered

tokens = tokenizer.encode("World")
print(tokens)
tokens = tokenizer.encode(" World")
print(tokens)
tokens = tokenizer.encode("  World")
print(tokens)
tokens = tokenizer.encode("   World")
print(tokens)


# Situationship is broken down into -> Sit + uations + hip

tokens = tokenizer.encode("Situationship")
print(tokens)
for token in tokens:
    print(tokenizer.decode([token]))

# Urbanisation -> Urban + isation

tokens = tokenizer.encode("Urbanisation")
print(tokens)
for token in tokens:
    print(tokenizer.decode([token]))


# Hello I am Aditya. I am from Patna, Bihar. I am learning how to create LLMs ->

tokens = tokenizer.encode("Hello I am Aditya. I am from Patna, Bihar. I am learning how to create LLMs")
print(tokens)
for token in tokens:
    print(tokenizer.decode([token]))


# Hello
#  I
#  am
#  Ad
# ity
# a
# .
#  I
#  am
#  from
#  Pat
# na
# ,
#  Bihar
# .
#  I
#  am
#  learning
#  how
#  to
#  create
#  LL
# Ms

# C++ code in tokenization

tokens = tokenizer.encode("int a = 2; int b = 3; cout<<(a+b)<<endl;")
print(tokens)
for token in tokens:
    print(tokenizer.decode([token]))
# int
#  a
#  =
#  2
# ;
#  int
#  b
#  =
#  3
# ;
#  cout
# <<
# (
# a
# +
# b
# )
# <<
# end
# l
# ;
