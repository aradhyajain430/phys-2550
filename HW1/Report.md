# PHYS 2550 - Homework 1

All code can be found at https://github.com/aradhyajain430/phys-2550/blob/master/HW1/answers.ipynb

## (a) Data

I opened the XLSX file in Excel and exported it as CSV.
The first 5 rows are printed in the output of the jupyter notebook.

| Column | Meaning | Units | Range |
|---|---|---|---|
| No | Row ID | None | 1 to 414 |
| X1 | Transaction date | Year/month code | 2012.667 to 2013.583 |
| X2 | House age | Years | 0 to 43.8 |
| X3 | Distance to nearest MRT station | Metres | 23.38284 to 6488.021 |
| X4 | Convenience stores within walking distance | Count | 0 to 10 |
| X5 | Latitude | Degrees north | 24.93207 to 25.01459 |
| X6 | Longitude | Degrees east | 121.47353 to 121.56627 |
| Y | Price per unit area | 10,000 NTD/ping | 7.6 to 117.5 |

There are 414 rows and no missing values. Most prices are well below the
highest value of 117.5, which stands out as an outlier. MRT distance has a
long right tail, reaching about 6.49 km. Distance is measured in thousands
of metres, while age only ranges from 0 to 43.8 years, so their scales are
very different.

## (b) Age only

Let $\hat y_i=w_0+w_1x_i$ and $r_i=w_0+w_1x_i-y_i$. The mean squared error is

$$L=\frac1n\sum_i r_i^2.$$

Differentiating gives

$$\frac{\partial L}{\partial w_0}
=\frac1n\sum_i2r_i\frac{\partial r_i}{\partial w_0}
=\frac2n\sum_i r_i,$$

$$\frac{\partial L}{\partial w_1}
=\frac1n\sum_i2r_i\frac{\partial r_i}{\partial w_1}
=\frac2n\sum_i r_ix_i.$$

I used a learning rate of **0.002** and **20,000 epochs**. In short trials (ran internally, code not given in the notebook), 0.002 reduced the loss faster than 0.001, while 0.003 made it increase. Hence, I decided to go with the learning rate used. The epoch count was the first of multiples of 5k epochs to converge.
The loss stopped changing to the reported precision during the last 1,000 epochs.

The final weights are $w_0=42.43469705$ and $w_1=-0.25148842$, with MSE **176.50047403**. Age alone is not a good predictor: its predictions stay in a narrow band even though the actual prices vary widely. The points are far from
the dashed line, especially for the cheapest and most expensive houses. This can be seen from the scatterplot - the diagonal line is what a good predictor should have, while the blue dots represent the scatter on prediction vs actual.

## (c) Age, MRT distance, and convenience stores

Using age, MRT distance, and store count as the features,

$$\mathbf x_i=(1,x_{i1},x_{i2},x_{i3})^T,\qquad
\hat y_i=\mathbf w^T\mathbf x_i,\qquad\hat{\mathbf y}=X\mathbf w.$$

The loss, gradient, and update are

$$L=\frac1n(X\mathbf w-\mathbf y)^T(X\mathbf w-\mathbf y),\qquad
\nabla L=\frac2nX^T(X\mathbf w-\mathbf y),$$

$$\mathbf w\leftarrow\mathbf w-\alpha\nabla L.$$

For the normalized run, each feature becomes $z_j=(x_j-\mu_j)/\sigma_j$.
The constant column and target stay unchanged. Both runs use a learning rate
of **0.05**. The raw run stops after **3 epochs** because its loss grows VERY rapidly.
The normalized run uses **600 epochs**.

The raw loss reaches about $1.42\times10^{35}$ after three updates.
Distances of thousands of metres produce large gradients, so the updates
overshoot. Normalization brings the feature scales closer together and lets
the same learning rate converge.

The final model, in normalized features, is

$$\hat y=37.98019324-2.87717494z_{\rm age}
-6.78084682z_{\rm distance}+3.81707864z_{\rm stores}.$$

Its MSE is **84.76070642**, about **52% lower** than the age-only loss.

## (d) Comparison

| Model | Final MSE |
|---|---:|
| Age only | 176.50047403 |
| Age + MRT distance + stores | 84.76070642 |

The model with more features follows the diagonal more closely. Distance and
store count add useful information, while age by itself is a weak predictor.
These fits do not test the other three features, so they do not show whether
those features would improve the model.The feature scale is what primarily caused the raw run to fail. Once the
features were normalized, the loss converged, so increasing the number of
epochs would not solve the remaining errors. Both models underpredict the
most expensive house, and squared loss is sensitive to this kind of outlier.

Next, I would inspect the outliers and residuals, try a log-distance or curved
age term, and compare the models on separate test data.

## Bonus: normal equation

Setting the gradient to zero gives

$$X^T(X\mathbf w-\mathbf y)=0
\quad\Longrightarrow\quad
(X^TX)\mathbf w=X^T\mathbf y
\quad\Longrightarrow\quad
\mathbf w_*=(X^TX)^{-1}X^T\mathbf y.$$

I use `np.linalg.solve` to solve the middle equation without forming the inverse.
Both methods use the same normalized features. The weights are close enough when
$\max_j|w_j-w_{*,j}|<10^{-6}$.

| Quantity | Value |
|---|---:|
| Gradient-descent MSE | 84.76070642 |
| Normal-equation MSE | 84.76070642 |
| Maximum coefficient difference at epoch 600 | $4.7\times10^{-11}$ |
| First epoch within tolerance | 352 |

Gradient descent reaches the chosen tolerance after **352 epochs**. At that
point, further updates do not change the MSE to six decimal places, so
$10^{-6}$ per coefficient is close enough for the reported results.


# AI Usage
AI was used for the following areas throughout the HW
(a): AI was used to generate the code for matplotlib as I personally don't know how to use it well
(b): AI was used to comment the written code throughout the submission
(c): AI was used to clean up and double check the language used in the actual report section of this HW.