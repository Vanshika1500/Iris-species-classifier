# iris classifier with visualization
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

st.title("🌸 Iris Species Classifier")

# 1. User input sliders
sepal_len = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_wid = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_len = st.slider("Petal length (cm)", 1.0, 7.0, 4.35)
petal_wid = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

# 2. Load data and train model
iris = load_iris()
X, y = iris.data, iris.target
model = SVC(probability=True)
model.fit(X, y)

# 3. Make prediction
sample = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
prediction = model.predict(sample)[0]
probs = model.predict_proba(sample)[0]

st.subheader("🔍 Prediction:")
st.write(f"**{iris.target_names[prediction].title()}**")

st.subheader("📊 Class Probabilities:")
for name, prob in zip(iris.target_names, probs):
    st.write(f"{name.title()}: {prob:.2f}")

# 4. Graphical Visualization
st.subheader("🌸 Your Input vs Dataset")
fig, ax = plt.subplots()
scatter = ax.scatter(
    iris.data[:, 2], iris.data[:, 3], c=y, cmap="viridis", edgecolor="k", s=50, alpha=0.7
)
ax.scatter(petal_len, petal_wid, color="red", s=100, label="Your Input", edgecolor="black")
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title("Petal Length vs Petal Width")
ax.legend()
st.pyplot(fig)


