# Mathematical Aspects of Machine Learning 


## Spring 2020
### Project Participants: Adriana Vega-Molino, Sean O'Hagan, Michael Gaiewski


\* All code is written in Python, unless otherwise stated. \*

We explore topics in image classification. Our first aim is to understand a Logistic Regression model and apply it on the famous MNIST dataset. Solving this classification problem allows us to predict labels based on images. The MNIST data consists of 70,000 examples of hand-written digits 0-9 along with their corresponding labels.

In the following, models are executed using only the training set (60,000 images) of the dataset found at [The MNIST Database](http://yann.lecun.com/exdb/mnist/).

{% gist 9ee980367fe890380be2fd3659a795b6 %}

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

<script src="https://gist.github.com/adrivega/9ee980367fe890380be2fd3659a795b6.js"></script>

You can use the [editor on GitHub](https://github.com/adrivega/adrivega.github.io/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

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
