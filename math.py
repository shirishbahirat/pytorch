

class mk:
   def __init__(self,val):
     self.val = val
   def __add__(self,other):
     out = mk(self.data + other.data, (self, other), '+')
     return out