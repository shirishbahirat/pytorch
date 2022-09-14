

class mk:
   def __init__(self,val):
     self.val = val

   def __add__(self,other):
     out = mk(self.val + other.val)
     return out

   def __mul__(self,other):
     out = mk(self.val * other.val)
     return out

   def __sub__(self,other):
     out = mk(self.val - other.val)
     return out

   def __div__(self,other):
     out = mk(self.val / other.val)
     return out

a = mk(3)
b = mk(4)

c = a + b
d = a * b
e = a - b
x = a / b

print(c.val)
print(d.val)
print(e.val)
print(x.val)
