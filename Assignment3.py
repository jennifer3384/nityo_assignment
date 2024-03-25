#!/usr/bin/env python
# coding: utf-8

# In[4]:


def sort_based_on_attribute():
    # Parsing the input
    # First line contains N and M
    N, M = map(int, input().split())
    athletes = []
    
    # Next N lines contain the athletes' details
    for _ in range(N):
        athletes.append(list(map(int, input().split())))
    
    # The last line contains K
    K = int(input())
    
    # Sorting the athletes based on the Kth attribute
    sorted_athletes = sorted(athletes, key=lambda x: x[K])
    
    # Printing the sorted list of athletes
    print("The sorted data is:")
    for athlete in sorted_athletes:
        print(' '.join(map(str, athlete)))


# In[5]:


#input the data, press enter to proceed to the next line
sort_based_on_attribute()


# In[ ]:




