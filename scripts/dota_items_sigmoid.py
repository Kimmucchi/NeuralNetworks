import pandas
import sklearn.model_selection as sk
import tensorflow as tf

# Importing the various collected datasets
players = pandas.read_csv("data/players.csv")
players2 = pandas.read_csv("data/players2.csv")
players3 = pandas.read_csv("data/players3.csv")
players4 = pandas.read_csv("data/players4.csv")
players5 = pandas.read_csv("data/players5.csv")
players6 = pandas.read_csv("data/players6.csv")
# Combining it all into one big dataset
players = pandas.concat([players,players2,players3,players4,players5,players6])

# Selecting out the columns of data we're going to use
players = players[['item_0', 'item_1','item_2','item_3','item_4','item_5','win']].dropna(axis=0)
print(players.shape)

# Separate out the features and the labels. Our features are the items. Our label is if they won or not.
features = players[['item_0', 'item_1','item_2','item_3','item_4','item_5']]
labels = players['win']
print(features.head())
print(labels.head())

# Splitting the data set into training (80%) and test (20%)
x_train, x_test, y_train, y_test = sk.train_test_split(features, labels, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)

# Setting up a neural network with 1 sigmoid activation function
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='sigmoid')
])

# Setting up the model's optimization strategy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model using 15 epochs
model.fit(x_train, y_train, epochs=15)

# Checking our model accuracy against the test data
print("*******************************")
print("\nEVALUATING TEST DATA")
print("*******************************")
model.evaluate(x_test, y_test)