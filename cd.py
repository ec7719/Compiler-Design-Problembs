#!/usr/bin/env python
# coding: utf-8

# In[83]:


# DAG
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px

graph = nx.DiGraph()

graph.add_node("a")
graph.add_node("b")
graph.add_node("c")
graph.add_node("d")

graph.add_edge("b", "a")
graph.add_edge("c", "a")
graph.add_edge("c", "b")
graph.add_edge("d", "b")

nx.draw(graph, with_labels=True)
plt.show()


# In[ ]:


# ##INFIX
from pythonds.basic.stack import Stack
def postfix_to_prefix(postfix_expr):
    stack = Stack()
    operators = set(['+', '-', '*', '/', '%', '^'])

    for token in postfix_expr.split():
        if token.isdigit():
            stack.push(token)
        elif token in operators:
            right = stack.pop()
            left = stack.pop()
            expr = token + left + right
            stack.push(expr)

    return stack.pop()

postfix_expr = "5 3 + 8 *"
prefix_expr = postfix_to_prefix(postfix_expr)
print(prefix_expr)  # Output: * + 5 3 8


# In[ ]:


from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

regex = Regex("(a|b)*abb")
nfa = regex.to_epsilon_nfa()
dfa = nfa.to_deterministic()

# Get the NFA expression
nfa_expr = nfa.to_regex()

print("dfa Expression: ", dfa)
print("NFA Expression: ", nfa_expr)


# In[54]:


from lark import Lark
# LR
grammar = """
    ?start: sum
    ?sum: product
         | sum "+" product  -> add
         | sum "-" product  -> subtract
    ?product: term
            | product "*" term  -> multiply
            | product "/" term  -> divide
    ?term: NUMBER            -> number
         | "-" term          -> negative
         | "(" sum ")"

    %import common.NUMBER
    %import common.WS_INLINE
    %ignore WS_INLINE
"""

parser = Lark(grammar)

expression = "5 * (4 + 3) - 2 / -1"

tree = parser.parse(expression)

print(tree.pretty())


# In[85]:


# #LEFT RECURSION
from pyparsing import *

A = Forward()
A <<= A + Literal('a') | Literal('b')

A <<= A + Literal('a') | Literal('b')
for i in range(10):
    A <<= (A + Literal('a')).setParseAction(lambda t: t[0][:-1]) + Literal('a') | Literal('b')

print(A)


# In[ ]:


from grako import parse
## LEading Trailing
grammar = """
    start = E;
    E = T ( '+' T )*;
    T = F ( '*' F )*;
    F = '(' E ')' | 'id';
"""


model = parse(grammar,"E")

model.compute_nullability()
model.compute_first_sets()
model.compute_follow_sets()

leading = {}
trailing = {}
for r in model.rules:
    leading[r.name] = r.first_set
    trailing[r.name] = r.follow_set

print(leading)
print(trailing)


# In[ ]:


import nltk
from nltk.tokenize import word_tokenize

# sample text
text = "The quick brown fox jumped over the lazy dog +."

# tokenize the text into words
words = word_tokenize(text)

# print the words
print(words)


# In[ ]:


class Quadruple:
    def __init__(self, op, arg1, arg2, result):
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2
        self.result = result

class Triple:
    def __init__(self, op, arg1, result):
        self.op = op
        self.arg1 = arg1
        self.result = result

class Double:
    def __init__(self, op, result):
        self.op = op
        self.result = result

# Get user input for quadruple
op = input("Enter operator: ")
arg1 = input("Enter argument 1: ")
arg2 = input("Enter argument 2: ")
result = input("Enter result: ")
q = Quadruple(op, arg1, arg2, result)

# Get user input for triple
op = input("Enter operator: ")
arg1 = input("Enter argument 1: ")
result = input("Enter result: ")
t = Triple(op, arg1, result)

# Get user input for double
op = input("Enter operator: ")
result = input("Enter result: ")
d = Double(op, result)

# Print the results
print("Quadruple: ", q.op, q.arg1, q.arg2, q.result)
print("Triple: ", t.op, t.arg1, t.result)
print("Double: ", d.op, d.result)


# # Lexical analysis

# In[ ]:


import nltk

# Define the keywords and operators
keywords = set(['if', 'else', 'while', 'for', 'return'])
operators = set(['+', '-', '*', '/', '=', '<', '>', '==', '!='])

# Open and read the file
with open('untitled4.txt', 'r') as file:
    text = file.read()

# Tokenize the text
tokens = nltk.word_tokenize(text)

# Filter the tokens to get punctuators, keywords, and operators
punctuators = [token for token in tokens if not token.isalnum()]
keywords = [token for token in tokens if token in keywords]
operators = [token for token in tokens if token in operators]

# Print the results
print('Punctuators:', punctuators)
print("\n ")
print('Keywords:', keywords)
print("\n ")
print('Operators:', operators)


# # First and Follow

# In[72]:


import nltk
from nltk import CFG

grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N
    VP -> V NP | V
    Det -> 'the'
    N -> 'cat' | 'dog'
    V -> 'chased' | 'ate'
""")

parser = nltk.ChartParser(grammar)

sentence = 'the cat chased the dog'.split()

for tree in parser.parse(sentence):
    print(tree)


# In[ ]:


from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton


# In[ ]:


regex=Regex("a|b*b")
nfa=regex.to_epsilon_nfa()
dfa=nfa.to_deterministic()
print("Nfa:",nfa.to_regex())
print("DFA:",dfa.to_regex())


# In[ ]:


from pythonds.basic.stack import Stack


# In[120]:


def postfix(expr):
    stack=Stack()
    operator=set(["+","-","*","/",".","^","(",")","%"])
    for i in expr.split():
        if i.isdigit():
            stack.push(i)
        elif i in operator:
            right=stack.pop()
            left=stack.pop()
            expression=i+right+left
            stack.push(expression)
    return stack.pop()
exp="5+9*1-5"
result=postfix(exp)
print(result)


# In[ ]:


import nltk 
with open ("untitled4.txt",'r') as file:
    text=file.read()
    


# In[ ]:


tokens=nltk.word_tokenize(text)


# In[ ]:


tokens


# In[ ]:


exp=[]
keywords = set(['if', 'else', 'while', 'for', 'return'])
operators = set(['+', '-', '*', '/', '=', '<', '>', '==', '!='])
for i in tokens:
    if i in keywords:
        exp=i
        print(exp)
    elif i in operators:
        exp=i
        print(exp)
    elif not i.isalnum():
        exp=i
        print(exp)   


# In[ ]:


exp


# In[ ]:


elif i in operators:
        exp=i
        print(exp)


# In[ ]:


class Quadruple():
    def __init__(self,op,arg1,arg2,result):
        self.op=op
        self.arg1=arg1
        self.arg2=arg2
        self.result=result
class Triplet():
    def __init__(self,op,arg1,result):
        self.op=op
        self.arg1=ar1
        self.result=result
class Doublet():
    def __init__(self,op,result):
        self.op=op
        self.result=result
        
op=input("Enter the operator")
arg1=input("Enter the operator")
arg2=input("Enter the operator")
result=input("Enter the operator")
q=Quadruple(op,arg1,arg2,result)
print(q.op,q.arg1,q.arg2,q.result)


# In[67]:


from lark import Lark


# In[73]:


from lark import Lark
# LR
grammar = """
    ?start: sum
    ?sum: product
         | sum "+" product  -> add
         | sum "-" product  -> subtract
    ?product: term
            | product "*" term  -> multiply
            | product "/" term  -> divide
    ?term: NUMBER            -> number
         | "-" term          -> negative
         | "(" sum ")"

    %import common.NUMBER
    %import common.WS_INLINE
    %ignore WS_INLINE
"""

parser = Lark(grammar)

expression = "5 * (4 + 3) - 2 / -1"

tree = parser.parse(expression)

print(tree.pretty())


# In[74]:


import nltk
from nltk import CFG


# In[78]:


grammar=CFG.fromstring("""
S -> NP VP | NP
NP -> "the" DEP
DEP -> "Lion"
""")
parser=nltk.ChartParser(grammar)
sentence="the Lion".split()
for i in parser.parse(sentence):
    print(i)


# In[82]:


import networkx as nx
import matplotlib.pyplot as plt
graph=nx.DiGraph()
graph.add_node("a")
graph.add_node("b")
graph.add_node("c")
graph.add_node("d")
graph.add_edge("c","b")
graph.add_edge("a","b")
graph.add_edge("a","d")
nx.draw(graph,with_labels=True)
plt.show()


# In[90]:


# #LEFT RECURSION
from pyparsing import *

A = Forward()
A <<= A + Literal('a') | Literal('b')

A <<= A + Literal('a') | Literal('b')
for i in range(10):
    A <<= (A + Literal('a')).setParseAction(lambda t: t[0][:-1]) + Literal('a') | Literal('b')

print(A)


# In[94]:


from pyparsing import *
A=Forward()
A<<=A+Literal('a')|Literal('b')
A<<=A+Literal('a')|Literal('b')
for i in range(10):
    A<<=(A+Literal('a')).setParseAction(lambda t:t[0][:-1])+Literal('a')|Literal('b')
print(A)


# In[99]:


from lark import Lark
grammar=""" 
    ?start: sum
    ?sum: product
         | sum "+" product  -> add
         | sum "-" product  -> subtract
    ?product: term
            | product "*" term  -> multiply
            | product "/" term  -> divide
    ?term: NUMBER            -> number
         | "-" term          -> negative
         | "(" sum ")"

    %import common.NUMBER
    %import common.WS_INLINE
    %ignore WS_INLINE
"""
parser=Lark(grammar)
arithmetic_expression="5+7/5"
result=praser.parse(arithmetic_expression)
print(result.pretty())


# In[103]:


from nltk import CFG
from nltk import ChartParser
grammar=CFG.fromstring(""" 
    S -> NP VP
    NP -> Det N
    VP -> V NP | V
    Det -> 'the'
    N -> 'cat' | 'dog'
    V -> 'chased' | 'ate'
""")
parser=ChartParser(grammar)
sentence="the dog chased the cat".split()
for i in parser.parse(sentence):
    print(i)


# In[118]:


import nltk
sentence="Luffy uussop is the best duo"
print(nltk.word_tokenize(sentence))
sent=nltk.word_tokenize(sentence)
for i in sent:
        
    if  i.isalnum():
        print(sentence)
    else:
        print("nami swann")


# In[112]:





# In[ ]:




