import streamlit as st
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("🌳 Decision Tree Hyperparameter Playground")

criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
splitter = st.sidebar.selectbox("Splitter", ["best", "random"])
max_depth = st.sidebar.slider("Max Depth", 1, 20, 3)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
max_features = st.sidebar.slider("Max Features", 1, 2, 2)

X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion=criterion,
                                splitter=splitter,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.markdown(f"### ✅ Accuracy Score: `{acc:.2f}`")


def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .01  

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Tree Classification Boundary")
    st.pyplot(plt)

plot_decision_boundary(model, X_test, y_test)


st.markdown("### 🌲 Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(15, 8))
plot_tree(model, filled=True, feature_names=["Feature1", "Feature2"], class_names=["Class 0", "Class 1"])
st.pyplot(fig)
