import pandas as pd
import matplotlib.pyplot as plt

path = './Documents/easyocr.csv'

data = pd.read_csv(path)

plt.plot(data['train_loss'], label='Train loss')
plt.plot(data['valid_loss'], label='Test loss')
plt.xlabel('Epoc (per 500)')
plt.ylabel('Value')
plt.title('Train & Test loss')
plt.legend()
plt.show()

# multiply norm_ED by 10
data['norm_ED'] = data['norm_ED'] * 100

plt.plot(data['acc'], label='Accuracy')
plt.plot(data['norm_ED'], label='Normalized Edit Distance')
plt.xlabel('Epoc (per 500)')
plt.ylabel('Value')
plt.title('Accuracy & Normalized Edit Distance')
plt.legend()
plt.show()