import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression



data = {
    'size' : [10,20,30,39],
    'prize' : [30,60,90,110]
}


df = pd.DataFrame(data)
print(df)


X=df[['size']]
y=df['prize']


model = LinearRegression()
model.fit(X,y)

y_pred=model.predict(X)



plt.scatter(X,y,color='red',label='size')
plt.plot(X,y_pred,color='blue',label='prize')
plt.title('house estimation model')
plt.xlabel('size / kanal')
plt.ylabel('prize / million')
plt.legend()
plt.show()


print(f"mean square error is: {mean_squared_error(y,y_pred)}")
print(f"slope is: {model.coef_}")
print(f"intercept is: {model.intercept_}")



new = np.array([[78]])
new_pred=model.predict(new)
print(f'estimated prize is {new_pred[0]:.2f} million')


