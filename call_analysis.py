import nltk
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

transcript_path = r'C:\Users\vanit\OneDrive\Desktop\Vanita\Workspace\ER\data.txt'
with open(transcript_path) as f:
    transript_data = f.read()
sentence =  transript_data.split("\n") 

tokenized_word=word_tokenize(transript_data)

stop_words=set(stopwords.words("english"))

filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)

fdist = FreqDist(filtered_sent)
fdist.plot(30,cumulative=False)
plt.show()

sentiment_call = TextBlob(transript_data)

sentiment_call.sentences
negative = 0
positive = 0
neutral = 0
all_sentences = []

for sentence in sentiment_call.sentences:
  print(sentence.sentiment.polarity)
  if sentence.sentiment.polarity < 0:
    negative +=1
  if sentence.sentiment.polarity > 0:
    positive += 1
  else:
    neutral += 1
 
  all_sentences.append(sentence.sentiment.polarity) 

all_sentences = np.array(all_sentences)
print('sentence polarity: ' + str(all_sentences.mean()))