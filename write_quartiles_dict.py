import sys


import pandas as pd

df = pd.read_csv('data/q.csv', error_bad_lines=False)

# print(list(df))

i=0
result = {}
for a in df.iterrows():

  i = i+1

  if (a[0] == '-;-'):
    continue

  _j = a[0].split(' ')

  if (len(_j) == 1):
    continue

  j = _j[1]
  b = j.split(';')

  # print(a[1][0])

  # print(j, len(b))

  if (len(b) > 1):
    #q in first line
    result[b[0].strip()] = b[1]

  else:
    #q in second line
    #two issns for one journal: i.e. online&offline
    c = a[1][0].split(';')
    result[b[0].strip()] = c[1]
    result[c[0].strip()] = c[1]

  sys.stdout.write('.')



import csv
w = csv.writer(open("data/q_dict.csv", "w"))
for key, val in result.items():
    w.writerow([key, val])

