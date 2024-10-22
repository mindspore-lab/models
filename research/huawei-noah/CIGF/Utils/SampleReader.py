import sys

def Read(filename, lines=20):
	ret = list()
	with open(filename, 'r', encoding='utf-8') as fs:
		cnt = 0
		for line in fs:
			cnt += 1
			print(line.strip())
			if cnt > lines:
				break
	return ret

if __name__ == "__main__":
	argLen = len(sys.argv)
	if argLen != 2 and argLen != 3:
		print('python SampleReader $filename [$lines=20]')
	elif argLen >= 3:
		arg2 = int(sys.argv[2])
		Read(sys.argv[1], arg2)
	else:
		Read(sys.argv[1])