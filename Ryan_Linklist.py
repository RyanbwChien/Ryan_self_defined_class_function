# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 21:42:06 2025

@author: user
"""

class Linklist:
    def __init__(self):
        self.node = None
        self.next = None
        self.prev = None
        
    def add(self, value):
        if self.node is None:
            self.node = value
        else:
            if self.next is None:
                self.next = Linklist()
                self.next.node = value
            else:
                self.next.add(value)
            
    def print_(self):
        if self.node is not None:
            print(self.node, end="->")
            if self.next is not None:
                self.next.print_()
    
    def reverse(self):
        prev = Linklist()
        prev.node = self.node
        current = self.next
        
        while current is not None:

                next_node = current.next                        
                current.next = prev
                prev = current
                current =  next_node
        return prev
        # self.node = prev.node
        # self.node = prev.node

# 0 is None
new = Linklist()
for i in range(10):
    new.add(i)
    
new.print_()

prev = new.reverse()
prev.print_()
