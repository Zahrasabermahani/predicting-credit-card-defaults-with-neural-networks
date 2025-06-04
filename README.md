# predicting-credit-card-defaults-with-neural-networks
A comparison of simple and deep neural networks to predict credit card default risk using customer attributes from the UCI dataset.


##  Predicting Credit Card Defaults – My Neural Network Tryout

For this small project, I used a real dataset of credit card users to see if I could predict who might default on a payment. It had 23 features per person — stuff like credit history, payment amounts, and more.

Let’s be honest, who wouldn't want to know ahead of time who might miss their payment?

---

###  What I did

- Pulled in the data with `numpy.genfromtxt()`  
- Scaled all the input features using `StandardScaler` so the model wouldn’t get confused by big number differences  
- Split the data into training and validation (60/40 felt fair)  
- Built two models:
  - **model_simple** – just one dense layer, like a fancy logistic regression  
  - **model_complex** – a deeper model with 5 layers, to see if depth actually helps

---

###  What I found

- Both models learned something, but the deeper one clearly picked up more patterns  
- Plotted accuracy over time – and yep, the complex one pulled ahead (especially on validation data)  
- Training was a bit slower with the deep model, but the payoff was better generalization

---

###  What I learned

Sometimes keeping it simple works… but when the data’s rich, going deeper gives better results. Just gotta watch out for overfitting and extra training time.

---

###  Tools I used

`numpy`, `scikit-learn`, `TensorFlow (Keras)`, `matplotlib`

