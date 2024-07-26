
idx1 = {}
idx2 = {}

for char in 'abcdefghijklmnopqrstuvwxyz':
  idx1[char] = []
  idx2[char] = []

with open('eng.dict', 'r', encoding='utf8') as f:
  for i, line in enumerate(f):
    if len(line) > 2:
      idx1[line[0].lower()].append(i)
      idx2[line[1].lower()].append(i)
  

with open('eng.idx1', 'w', encoding='utf8') as f:
  for char in idx1:
    if len(idx1[char]) > 0:
      f.write(char + ' ' + str(idx1[char])[1:-1].replace(' ', '') + '\n')

with open('eng.idx2', 'w', encoding='utf8') as f:
  for char in idx2:
    if len(idx2[char]) > 0:
      f.write(char + ' ' + str(idx2[char])[1:-1].replace(' ', '') + '\n')
