import numpy as np
from google.colab import files
from tensorflow.keras.utils import load_img, img_to_array

uploaded = files.upload()

for fn in uploaded.keys():
  # 1. Load the image
  path = '/content/' + fn
  img = load_img(path, target_size=(150, 150))

  # 2. Convert to Array
  x = img_to_array(img)

  # --- THE FIX: Normalize the data to match training (0 to 1) ---
  x /= 255.0
  # --------------------------------------------------------------

  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])

  # 3. Predict
  classes = model.predict(images, batch_size=10)

  # Print the actual probability number to see how confident it is
  print(f"Raw Probability: {classes[0][0]:.4f}")

  if classes[0]>0.5:
    print(f"result: {fn} is a DOG")
  else:
    print(f"result: {fn} is a CAT")
