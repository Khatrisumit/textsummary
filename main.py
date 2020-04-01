
# coding=utf-8
from flask import Flask, url_for, render_template, request, jsonify, make_response
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from gensim.summarization import summarize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import readtime
import os


app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
@app.route('/index', methods = ['GET', 'POST'])
def index():
	if(request.method == 'GET'):
            return render_template('index.html',status=False)
	else:
            doc = request.form['text']
            nltk.data.path.append(os.path.join(app.root_path,'nltk_data'))
            nltkSummary = nltkSummarize(doc)
            spacySummry = spacySummarize(doc)
            gensimSummry = gensimSummarize(doc)
            if(spacySummry == False or gensimSummry == False):
                return render_template('index.html', status=False, message="Input is too short for summarization")
                #return 'false'
            else:
                return render_template(
                    'index.html',
                    data={
                        'spacy' : {
                            'summary' : spacySummry,
                            'time' : readtime.of_text(spacySummry)
                        },
                        'nltk' : {
                            'summary' : nltkSummary,
                            'time' : readtime.of_text(nltkSummary)
                        },
                        'gensim' : {
                            'summary' : gensimSummry,
                            'time' : readtime.of_text(gensimSummry)
                        },
                        'original_time' : readtime.of_text(doc),
                        'input' : doc
                    }
                )
		

def spacySummarize(text):
	extra_words=list(STOP_WORDS)+list(punctuation)+['\n']
	nlp=spacy.load('en_core_web_sm')

	docx = nlp(text)

	all_words=[word.text for word in docx]
	Freq_word={}
	for w in all_words:
		w1=w.lower()
		if w1 not in extra_words and w1.isalpha():
			if w1 in Freq_word.keys():
				Freq_word[w1]+=1
			else:
				Freq_word[w1]=1
				  
	val=sorted(Freq_word.values())
	max_freq=val[-3:]
	topic = []
	print("Topic of document given :-")
	for word,freq in Freq_word.items():
		if freq in max_freq:
			topic.append(word)
		else:
			continue
			  
	for word in Freq_word.keys():
	   Freq_word[word] = (Freq_word[word]/max_freq[-1])
	
	sent_strength={}
	for sent in docx.sents:
		for word in sent :
			if word.text.lower() in Freq_word.keys():
				if sent in sent_strength.keys():
					sent_strength[sent]+=Freq_word[word.text.lower()]
				else:
					sent_strength[sent]=Freq_word[word.text.lower()]
			else: 
				continue
	top_sentences=(sorted(sent_strength.values())[::-1])
	top30percent_sentence=int(0.15*len(top_sentences))
	top_sent=top_sentences[:top30percent_sentence]
	
	
	summary=[]
	for sent,strength in sent_strength.items():
		if strength in top_sent:
			 summary.append(sent)
		else:
		   continue
	
	if(len(summary) == 0):
		return False
	else:
		final_summary = ' '.join([str(paragraph) for paragraph in summary]) 
		return final_summary
	
def gensimSummarize(doc):
	try :
		summary =summarize(doc, ratio=0.2, word_count=None, split=False)
		return summary
	except Exception as e:
		return False
	

def nltkSummarize(doc):
	freq_table = _create_frequency_table(doc)

	'''
	We already have a sentence tokenizer, so we just need 
	to run the sent_tokenize() method to create the array of sentences.
	'''
	
	# 2 Tokenize the sentences
	sentences = sent_tokenize(doc)
	
	# 3 Important Algorithm: score the sentences
	sentence_scores = _score_sentences(sentences, freq_table)
	
	# 4 Find the threshold
	threshold = _find_average_score(sentence_scores)
	
	# 5 Important Algorithm: Generate the summary
	summary = _generate_summary(sentences, sentence_scores, 1 * threshold)
	
	print('NLTK Summary')
	print(summary)
	return summary

def _create_frequency_table(text_string) -> dict:

    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable



def _score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue

def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
