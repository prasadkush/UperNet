

def find_nth_occurence(str, ch, n=1):
	count = 0
	ind = -1
	index = -1
	#print('inside find_nth_occurence')
	#print('n: ', n, 'ind: ', ind)
	while count <= n and index < len(str) - 1:
		#print('inside while')
		ind = str[index+1:].find(ch)
		#print('ind: ', ind)
		if ind != -1:
			index += ind + 1
			count += 1
		else:
			index = len(str) + 1
		#print('ind: ', ind)
		#print('index: ', index)
	if index == -1:
		index = len(str) + 1
	#print('index: ', index)
	return index

params_to_print = ['swintransformer.stageblock1.transformerblocks.0.msablock.linear1.weight', 'swintransformer.stageblock2.transformerblocks.1.mlpblock.mlplayer.3.weight', 'swintransformer.stageblock3.transformerblocks.2.msablock.linear1.weight', 'swintransformer.stageblock4.transformerblocks.1.mlpblock.mlplayer.3.weight', 'crf1.layers.0.mlpblock.mlplayer.0.weight', 'crf2.layers.0.mlpblock.mlplayer.3.weight', 'crf3.layers.0.mlpblock.mlplayer.0.weight', 'crf4.layers.0.mlpblock.mlplayer.3.weight']
paramnames = [p[0:min(find_nth_occurence(p,'.',n=6),len(p))] for p in params_to_print]
print('paramnames: ', paramnames, '\n')


