import pandas as pd
import split_module
import spellcheck_module

class InputPipeline:
	test_dictionary = []
	def __init__(self):
		print("Created Input Pipeline")
		self.import_dictionary()

	def import_dictionary(self):
		self.test_dictionary = pd.read_csv("suggestions_cleared.csv",header=0)

	def split_text(self,input):
		output_list = []
		cleaned_review = split_module.clean_review(input)
		tokenized_sentences = split_module.tokenize_review(cleaned_review)
		for sentence in tokenized_sentences:
			sentence = ' '.join(sentence)
			if (split_module.text_filter(sentence)):
				pro_cons_list = split_module.split_pros_cons(sentence)
				for i in pro_cons_list:
					if(split_module.text_filter(i)):		
						output_list.append(i)
		return output_list

	def spellcheck(self,sentence):
		word_list = sentence.split()
		new_word_list = []
		for word in word_list:
			lower_word = word.lower()
			lower_word = spellcheck_module.check_dictionary(lower_word,self.test_dictionary)
			new_word_list.append(lower_word)
		new_word = " ".join(new_word_list)
		return new_word

	def print_list(self,input_list):
		for i in input_list:
			print("~~~~~")
			print(i)
			print("~~~~~")

	def pipeline(self,text):
		text_split = self.split_text(text)
		spellchecked_list = []
		for sentence in text_split:
			spellchecked_list.append(self.spellcheck(sentence))
		# self.print_list(spellchecked_list)
		return spellchecked_list

# input_pipeline = InputPipeline()
# input_pipeline.pipeline(text)