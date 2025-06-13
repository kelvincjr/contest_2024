# -*- coding:utf-8 -*-

import os,jpype

os.environ['JAVA_HOME']="/bdci_2024/jdk1.8.0_212-amd64"
current_dir = os.path.dirname(os.path.abspath(__file__))

#jarpath = os.path.join(os.path.abspath(r'./trib_lib'), 'tree_split-1.3.jar')
jarpath = os.path.join(current_dir, 'trie_lib/tree_split-1.3.jar')
print("current_dir {}, jarpath {}".format(current_dir, jarpath))
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % jarpath)

Forest = jpype.JClass('love.cq.domain.Forest')
Library = jpype.JClass('love.cq.library.Library')
GetWord = jpype.JClass('love.cq.splitWord.GetWord')

print("trie_lib init")

class trie_lib:
	def __init__(self):
		self.forest = Forest()
	#word_list = ['前埔五里', '洪文五里', '前埔东一里', '洪文二里', '洪莲西二路', '洪文石村社北片区', '前埔南路', '会展南路', '侨兴里', '文兴东三里', '洪文石村社南片区', '洪莲东二里', '洪文三里', '洪文泥窟社南片区', '洪莲中路', '前埔一里', '洪莲里', '东山社', '店上西里', '古楼北里', '文兴东一里', '侨文里', '东浦三里', '西林东里', '古楼南里', '会展北路', '前埔西路', '洪莲西二里', '洪莲中二路', '西林路', '侨龙里', '洪文七里', '西林西二路', '洪莲北路', '东坪山社', '洪莲西里', '西林西二里', '前埔中路', '前埔六里', '洪文六里', '洪莲路', '东浦路', '西林东路', '洪文四里', '东芳山庄', '洪文八里', '云顶岩路', '金尚路', '店上东里', '侨洪里', '文兴东二里', '田厝路', '吕岭路', '前埔二里', '莲前西路', '云顶中路', '莲前东路', '洪文一里', '西林西里', '洪文泥窟社北片区', '西林社', '洪莲西路']

	def build_trie(self, word_list):
		for w in word_list:
			Library.insertWord(self.forest, w)

	#test_str = "洪莲西里洪山柄北区71#502室"

	def trie_search(self, search_str):
		temp_str = search_str
		match_words = []
		while temp_str is not None:
			udg = self.forest.getWord(temp_str)
			match_word = udg.getAllWords()
			if match_word is None:
				break
			match_word = str(match_word)
			match_words.append(match_word)
			pos = temp_str.find(match_word)
			if pos is None:
				break
			temp_str = temp_str[pos+len(match_word):]
			#print('match_word: {}, temp_str: {}'.format(match_word, temp_str))
		return match_words

	def close_jvm(self):
		jpype.shutdownJVM()