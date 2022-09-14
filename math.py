

class mk:
   def __init__(self,val):
     self.val = val
   def __add__(self,other):
     out = mk(self.val + other.val, (self, other), '+')
     return out


a = mk(3)
b = mk(4)

c = a + b
