import flask
from flask import Flask
import os
import random
import tensorflow as tf
import numpy as np

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model = tf.keras.models.load_model('saved_model/my_model')
model.compile(optimizer='adam', loss=loss)

corpus = open('data.txt', encoding='utf-8').read()
vocab = sorted(set(corpus))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def generate_text(model, start_string):
  # evaluation step (generating text using the learned model)

  # number of characters to generate
  num_generate = 1500

  # converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # empty string to store our results
  text_generated = []

  # low temperatures results in more predictable text.
  # higher temperatures results in more surprising text.
  # :D
  temperature = 1.0

  # here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method ==  'POST':
        starter_list = ['We ', 'The ', 'Although ', 'During ', 'With ', 'NYU ']
        starting_word = random.choice(starter_list)
        
        email_contents = generate_text(model, starting_word)
        print("something worked")
        print(email_contents)

        return flask.render_template('main.html', result=email_contents,)



if __name__ == '__main__':
    app.run(threaded=True, port=5000)
