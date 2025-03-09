#get the set of characters in a language

meta = 'onespeaker.csv'

f = open(meta,'r')
t = f.read()
f.close()

t = t.split('\n')

chars = set()
for line in t:
	bits = line.split('|')
	if len(bits) > 1:
		for char in bits[1]:
			chars.add(char)

print(''.join(chars))

