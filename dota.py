import pandas
import sklearn.model_selection as sk
import tensorflow as tf

players = pandas.read_csv("scripts/players.csv")
players2 = pandas.read_csv("scripts/players2.csv")
players3 = pandas.read_csv("scripts/players3.csv")
players4 = pandas.read_csv("scripts/players4.csv")
players5 = pandas.read_csv("scripts/players5.csv")
players6 = pandas.read_csv("scripts/players6.csv")
players = pandas.concat([players,players2,players3,players4,players5,players6])
#players = players[['gold_per_min', 'hero_damage', 'hero_healing', 'level', 'kills', 'deaths', 'assists', 'win']].dropna(axis=0)
#players = players[['item_0', 'item_1','item_2','item_3','item_4','item_5','win']].dropna(axis=0)
#players = players[['hero_id','win']].dropna(axis=0)
players = players[['gold_per_min', 'hero_damage', 'hero_healing', 'level', 'kills', 'deaths', 'assists', 'item_0', 'item_1','item_2','item_3','item_4','item_5','hero_id','win']].dropna(axis=0)
print(players.shape)
print(players.sample(n=5))
# Separate out the features and the labels. Our only feature right now is the hero_id. Our label is if they won or not
features = players[['gold_per_min', 'hero_damage', 'hero_healing', 'level', 'kills', 'deaths', 'assists', 'item_0', 'item_1','item_2','item_3','item_4','item_5','hero_id']]
labels = players['win']

print(features.head())
print(labels.head())

# Let's split this data set into training and test
x_train, x_test, y_train, y_test = sk.train_test_split(features, labels, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)
print("*******************************")
print("\nEVALUATING TEST DATA")
print("*******************************")
model.evaluate(x_test, y_test)