f = open('__dump.txt','r')
f1 = open('__c1.txt', 'w')
f2 = open('__c2.txt', 'w')
f3 = open('__c3.txt', 'w')

for row in f:
  v = row.split(' ')
  if int(v[1]) > 21 and int(v[1]) < 40:
    f1.write(row)
  elif int(v[1]) > 40:
    f2.write(row)
  else:
    f3.write(row)

f.close()
f1.close()
f2.close()
f3.close()