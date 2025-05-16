import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
from nltk.stem.porter import * #added
import string #added
from nltk.stem.snowball import SnowballStemmer

#from nltk.stem.snowball import SnowballStemmer #added

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text) #得到文本中的詞彙列表
    tokens = [token.strip() for token in tokens] #對分詞後的詞彙列表進行去除空格的處理，確保每個詞彙的內容沒有前後的空格
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list] #保留不在停用詞列表中的詞彙(轉換成小寫後再檢查)
    preprocessed_text = ' '.join(filtered_tokens) #新組合成一個字符串，每個詞彙之間以空格分隔

    return preprocessed_text


def preprocessing_function(text: str) -> str: #take str[] as input and return a str[]
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    # Stemming
    preprocessed_text = preprocessed_text.lower()
    preprocessed_text = preprocessed_text.replace("<br / >", " ") # remove useless char with spaces

    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    #print(stemmer.stem('working'))

    #remove all the punctuation characters
    preprocessed_text = "".join([char for char in preprocessed_text if (char not in string.punctuation)]) # Remove punctuations

    preprocessed_text = [stemmer.stem(word) for word in preprocessed_text.split()]
    # End your code

    return preprocessed_text

'''
str4 = "So im not a big fan of Boll's work but then again not many are. I enjoyed his movie Postal (maybe im the only one). Boll apparently bought the rights to use Far Cry long ago even before the game itself was even finsished. <br /><br />People who have enjoyed killing mercs and infiltrating secret research labs located on a tropical island should be warned, that this is not Far Cry... This is something Mr Boll have schemed together along with his legion of schmucks.. Feeling loneley on the set Mr Boll invites three of his countrymen to play with. These players go by the names of Til Schweiger, Udo Kier and Ralf Moeller.<br /><br />Three names that actually have made them selfs pretty big in the movie biz. So the tale goes like this, Jack Carver played by Til Schweiger (yes Carver is German all hail the bratwurst eating dudes!!) However I find that Tils acting in this movie is pretty badass.. People have complained about how he's not really staying true to the whole Carver agenda but we only saw carver in a first person perspective so we don't really know what he looked like when he was kicking a**.. <br /><br />However, the storyline in this film is beyond demented. We see the evil mad scientist Dr. Krieger played by Udo Kier, making Genetically-Mutated-soldiers or GMS as they are called. Performing his top-secret research on an island that reminds me of SPOILER Vancouver for some reason. Thats right no palm trees here. Instead we got some nice rich lumberjack-woods. We haven't even gone FAR before I started to CRY (mehehe) I cannot go on any more.. If you wanna stay true to Bolls shenanigans then go and see this movie you will not be disappointed it delivers the true Boll experience, meaning most of it will suck.<br /><br />There are some things worth mentioning that would imply that Boll did a good work on some areas of the film such as some nice boat and fighting scenes. Until the whole cromed/albino GMS squad enters the scene and everything just makes me laugh.. The movie Far Cry reeks of scheisse (that's poop for you simpletons) from a fa,r if you wanna take a wiff go ahead.. BTW Carver gets a very annoying sidekick who makes you wanna shoot him the first three minutes he's on screen."
str3 = "I saw this movie when I was about 12 when it came out. I recall the scariest scene was the big bird eating men dangling helplessly from parachutes right out of the air. The horror. The horror.<br /><br />As a young kid going to these cheesy B films on Saturday afternoons, I still was tired of the formula for these monster type movies that usually included the hero, a beautiful woman who might be the daughter of a professor and a happy resolution when the monster died in the end. I didn't care much for the romantic angle as a 12 year old and the predictable plots. I love them now for the unintentional humor.<br /><br />But, about a year or so later, I saw Psycho when it came out and I loved that the star, Janet Leigh, was bumped off early in the film. I sat up and took notice at that point. Since screenwriters are making up the story, make it up to be as scary as possible and not from a well-worn formula. There are no rules."
str1 = "Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />This movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.<br /><br />OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them."
str2 = "Petter Mattei's Love in the Time of Money is a visually stunning film to watch. Mr. Mattei offers us a vivid portrait about human relations. This is a movie that seems to be telling us what money, power and success do to people in the different situations we encounter. <br /><br />This being a variation on the Arthur Schnitzler's play about the same theme, the director transfers the action to the present time New York where all these different characters meet and connect. Each one is connected in one way, or another to the next person, but no one seems to know the previous point of contact. Stylishly, the film has a sophisticated luxurious look. We are taken to see how these people live and the world they live in their own habitat.<br /><br />The only thing one gets out of all these souls in the picture is the different stages of loneliness each one inhabits. A big city is not exactly the best place in which human relations find sincere fulfillment, as one discerns is the case with most of the people we encounter.<br /><br />The acting is good under Mr. Mattei's direction. Steve Buscemi, Rosario Dawson, Carol Kane, Michael Imperioli, Adrian Grenier, and the rest of the talented cast, make these characters come alive.<br /><br />We wish Mr. Mattei good luck and await anxiously for his next work."
str5 = "Some films just simply should not be remade. This is one of them. In and of itself it is not a bad film. But it fails to capture the flavor and the terror of the 1963 film of the same title. Liam Neeson was excellent as he always is, and most of the cast holds up, with the exception of Owen Wilson, who just did not bring the right feel to the character of Luke. But the major fault with this version is that it strayed too far from the Shirley Jackson story in it's attempts to be grandiose and lost some of the thrill of the earlier film in a trade off for snazzier special effects. Again I will say that in and of itself it is not a bad film. But you will enjoy the friction of terror in the older version much more."
newstr = preprocessing_function(str2)
print(newstr)
'''