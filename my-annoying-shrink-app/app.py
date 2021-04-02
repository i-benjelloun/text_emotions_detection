# import libraries
import numpy as np
from transformers import TFBertModel, BertTokenizerFast, BertConfig
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal
import streamlit as st
from PIL import Image
import myfuncs

# Setting the taxonomy (including 'Neutral')
GE_taxonomy = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
                "confusion", "curiosity", "desire", "disappointment", "disapproval", 
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
                "joy", "love", "nervousness", "optimism", "pride", "realization", 
                "relief", "remorse", "sadness", "surprise", "neutral"]

# Dictionary of emotions / Emoji / responses
mapping_emotions = {
  "admiration":["&#x1F929;","\"You always admire what you really don't understand.\""],
  "amusement":["&#x1F973;","\"The unfortunate who has to travel for amusement lacks capacity for amusement.\""], 
  "anger":["&#x1F621;","\"Bitterness is like cancer. It eats upon the host. But anger is like fire. It burns it all clean.\""], 
  "annoyance":["&#x1F624;","\"It could be worse.\""], 
  "approval":["&#x1F44D;","\"A man cannot be confortable without his own approval.\""], 
  "caring":["&#x1F618;","\"The road to hell is paved with good intentions.\""], 
  "confusion":["&#x1F928;","\"It's ok to be confused, but the most important thing is to know why...\""], 
  "curiosity":["&#x1F914;","\"The cure for boredom is curiosity. The is no cure for curiosity.\""], 
  "desire":["&#x1F924;","\"From the deepest desire often come the deadliest hate.\""], 
  "disappointment":["&#x1F614;","\"You know the grass is always greener on the other side of the fence.\""], 
  "disapproval":["&#x1F644;","\"Don't judge a book by its cover.\""], 
  "disgust":["&#x1F92E;","\"Some people are just beautifully wrapped boxes of s***.\""], 
  "embarrassment":["&#x1F633;","\"Shame is a soul eating emotion.\""], 
  "excitement":["&#x1F606;","\"Excitement always leads to tears.\""], 
  "fear":["&#x1F631;","\"Limits, like fears, are often just an illusion.\""], 
  "gratitude":["&#x1F64F;","\"Revenge is profitable, gratitude is expensive.\""], 
  "grief":["&#x1F940;","\"All good things come to an end.\""], 
  "joy":["&#x1F917;","\"Ok then, everything is fine. It's 50$ please.\""],
  "love":["&#x1F60D;","\"Love turns, with a little indulgence, to indifference or disgust; hatred alone is immortal.\""], 
  "nervousness":["&#x1F62C;","\"I wish my metabolism worked as fast as your anxiety.\""], 
  "optimism":["&#x1F4AA;","\"Hope for the best, prepare for the worst.\""], 
  "pride":["&#x1F60E;","\"Among the blind the one eyed man is the king.\""], 
  "realization":["&#x1F947;","\"Don’t blow your own trumpet.\""], 
  "relief":["&#x1F62A;","\"Phewwww &#x1F605;\""], 
  "remorse":["&#x1F61E;","\"You can’t unscramble a scrambled egg.\""], 
  "sadness":["&#x1F62D;","\"Don't worry, a couple of drinks and you will be just fine.\""], 
  "surprise":["&#x1F62E;","\"Sometimes I drink water just to surprise my liver.\""], 
  "neutral":["&#x1F636;","So ??? &#x1F914;"]
  }



# Define a function that detects emotions in a text sample using a fine-tuned BERT model
# This function returns detected emotions (labels), associated probabilites for each label and the label with highest probability
def predict_sample(text_sample, model, tokenizer, threshold=0.87):
  
  # Clean text
  text_sample = myfuncs.preprocess_corpus(text_sample)  

  # Tokenize text
  sample = myfuncs.tokenize(tokenizer, text_sample)
  
  # Probability predictions
  sample_probas = model.predict(sample)
  sample_probas = sample_probas.ravel().tolist()

  # Label prediction using threshold
  sample_labels = [1 if (p>threshold) else 0 for p in sample_probas]
  best_proba = np.argmax(sample_probas)
  best_label = GE_taxonomy[best_proba]

  # Retrieving emotion names
  sample_labels = [GE_taxonomy[i] for i in range(len(sample_labels)) if sample_labels[i]==1]
  
  # Keeping only non-null probabilities
  sample_probas = [p for p in sample_probas if (p>threshold)]

  # Neutral if no emotion detected
  if len(sample_labels)==0:
    sample_labels.append("neutral")
    sample_probas.append("-")
    best_label = "neutral"

  return sample_labels, sample_probas, best_label



if __name__ == "__main__":

  # Loading BERT base model configuration
  config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=False)

  # Loading BERT tokenizer
  tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = 'bert-base-uncased', config = config)

  # Function to create a BERT based model
  @st.cache()
  def create_model(nb_labels, max_length=48, model_name='bert-base-uncased'):

    # Loading BERT main layer
    transformer_model = TFBertModel.from_pretrained(model_name, config = config)
    bert = transformer_model.layers[0]

    # Build the model inputs
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
    token_ids = Input(shape=(max_length,), name='token_ids', dtype='int32')
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_ids': token_ids}

    # Load the Transformers BERT model as a layer in a Keras model
    bert_model = bert(inputs)[1]
    dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)

    # Then build your model output
    emotion = Dense(units=nb_labels, activation="sigmoid", kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='emotion')(pooled_output)
    outputs = emotion

    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel')

   # return model
    return model

  # Load weights to model
  @st.cache()
  def load_all(model, weights):
    model.load_weights(weights)
    return model

  # Create a BERT-based model for multi-label classification (27 labels)
  model = create_model(27)

  # Load pre-trained model weights
  model = load_all(model, "../bert-weights.hdf5")
  
  # Web page layout elements
  st.markdown("<style>h1{text-align: right ; color: #2AA78D ; font-size:70px;}</style>", unsafe_allow_html=True)
  st.markdown("<style>.project{text-align: center ; color: #2AA78D ;}</style>", unsafe_allow_html=True)
  st.markdown("<style>.names{text-align: right ; }</style>", unsafe_allow_html=True)
  st.markdown("<style>.output{color: #2AA78D ;}</style>", unsafe_allow_html=True)
  st.markdown("<style>.solution{font-size:1.5em ; font-weight: lighter ;}</style>", unsafe_allow_html=True)
  st.markdown("<style>.space_col2{font-size:7px ;}</style>", unsafe_allow_html=True)
  
  # Header
  image = Image.open('images/image_psy.jpeg')
  header_col1, header_col2 = st.beta_columns(2)
  header_col1.markdown("<h2>""</h2>", unsafe_allow_html=True)
  header_col1.image(image)
  header_col2.markdown("<h1>My Annoying Shrink</h1>", unsafe_allow_html=True)
  
  for i in range(2):
    st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("<h2>How are you feeling today ?</h2>", unsafe_allow_html=True)

  
  # Enter text, detect emotions and display emojis, probabilities and answers
  user_text = st.text_input("")

  if user_text:

    # Button to display detected emotions
    mybutton = st.button("Display emotions")

    # Retrieve emotions, probabilties and the best label
    emotions, probabilities, best_label = predict_sample(user_text, model, tokenizer, 0.92)

    # Create a dictionary emotions/probabilities and sort by descending probabilities
    dico = {e:p for e,p in zip(emotions, probabilities)}
    dico = {k: v for k, v in sorted(dico.items(), key=lambda item: item[1], reverse=True)}
  
    # Dislplay answers according to detected emotions
    st.markdown("<h2 class='output'>Our expert's answer : </h2>", unsafe_allow_html=True)
    st.markdown("<p class='solution'>{}<p>".format(mapping_emotions[best_label][1]), unsafe_allow_html=True)

    # If the button has been clicked, display emotions and probabilities
    if mybutton:
      st.markdown("<h2 class='output'>Emotion(s) detected: </h2>", unsafe_allow_html=True)

      # If the user text is neutral, do not display a probability
      if best_label == "neutral":
        st.markdown("<h2>{} {}</h2>".format(mapping_emotions[best_label][0], best_label), unsafe_allow_html=True)

      else:
        for emo, prob in dico.items():
          st.markdown("<h2>{} {} ({:.2f}%)</h2>".format(mapping_emotions[emo][0], emo, prob*100), unsafe_allow_html=True)

  # Footer
  for i in range(5):
    st.markdown("<br>", unsafe_allow_html=True)

  logo = Image.open('images/logo_jedha.jpeg')
  footer_col1, footer_col2, footer_col3 = st.beta_columns(3)
  footer_col1.image(logo, width=120)

  footer_col2.markdown("<p></p>", unsafe_allow_html=True)
  footer_col2.markdown("<h2 class='project'>Jedha Fullstack Data Science Project</h1>", unsafe_allow_html=True)

  footer_col3.markdown("<p class='names'>Perrine Panisset</p>", unsafe_allow_html=True)
  footer_col3.markdown("<p class='names'>Ibrahim Benjelloun</p>", unsafe_allow_html=True)
  footer_col3.markdown("<p class='names'>Florian Akretche</p>", unsafe_allow_html=True)
  

