# Mathematical Aspects of Machine Learning 


## Spring 2020
### Project Participants: Adriana Vega-Molino, Sean O'Hagan, Michael Gaiewski


\* All code is written in Python, unless otherwise stated. \*

We explore topics in image classification. Our first aim is to understand a Logistic Regression model and apply it on the famous MNIST dataset. Solving this classification problem allows us to predict labels based on images. The MNIST data consists of 70,000 examples of hand-written digits 0-9 along with their corresponding labels.

In the following, models are executed using only the training set (60,000 images) of the dataset found at [The MNIST Database](http://yann.lecun.com/exdb/mnist/).

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("MNIST_data/train.csv")

x = data[data.columns[1:]]
y = data.label
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(data = scaler.transform(X_train) , columns = X_train.columns , index = X_train.index)
X_test = pd.DataFrame(data = scaler.transform(X_test), columns  = X_test.columns)

model = LogisticRegression(C=100, solver = 'lbfgs', max_iter = 1000, multi_class='multinomial')
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(accuracy)
0.9003968253968254
```

To develop a deeper understanding of the mathematics behind a Logistic Regression classifier, we resort to Charles Elkan's lecture *Maximum Likelihood, Logistic Regression, and Stochastic Gradient Training*. The logistic regression model is detailed as <img src="https://render.githubusercontent.com/render/math?math=log%20%5Cfrac%7Bp(x)%7D%7B1-p(x)%7D%20%3D%20%5Calpha%20%2Bx%20%5Ccirc%20%5Cbeta">
 and solving for the predicted probability, obtain <img src="https://render.githubusercontent.com/render/math?math=p(x)%3D%5Cfrac%7B1%7D%7B1%2B%5Cexp-(%5Calpha%20%2B%20x%5Ccirc%20%5Cbeta)%7D">. The model is able to learn by minimizing the log conditional likelihood function via the method of Stochastic Gradient Descent providing us, in turn, with an update formula <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta%20%3D%20%5Calpha%20%2B%20%5Clambda%20(y-p)x"> .

<!--  img commment here src="https://render.githubusercontent.com/render/math?math=log%20%5Cfrac%7Bp(x)%7D%7B1-p(x)%7D%20%3D%20%5Calpha%20%2Bx%20%5Ccirc%20%5Cbeta" -->

<!--  https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b  -->


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/adrivega/adrivega.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
