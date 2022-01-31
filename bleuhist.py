import sys
import matplotlib.pyplot as plt

bleuf = open(sys.argv[1], 'r')

x = list()

for line in bleuf:
    bleus = line.split('\t')
    
    Bas = bleus[2]
    
    x.append(float(Bas))

plt.hist(x, bins=20)
plt.show()
