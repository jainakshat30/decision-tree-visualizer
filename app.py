import streamlit as st
from streamlit import cache_data, cache_resource
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd  # Add this import at the top if not already present
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

st.title("🧠 ML Algorithm Visualizer Playground")

model_name = st.sidebar.selectbox("Select Algorithm", ["Decision Tree", "Random Forest", "Logistic Regression", "K-Nearest Neighbors"])

criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
splitter = st.sidebar.selectbox("Splitter", ["best", "random"])
max_depth = st.sidebar.slider("Max Depth", 1, 20, 3)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
max_features = st.sidebar.slider("Max Features", 1, 2, 2)

dataset_name = st.sidebar.selectbox("Select Dataset", ["Classification (2D Toy)", "Iris", "Wine", "Breast Cancer", "Upload Your Own"])

user_file = None
if dataset_name == "Upload Your Own":
    user_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

@st.cache_data
def get_dataset(dataset_name, user_file):
    if dataset_name == "Classification (2D Toy)":
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
        feature_names = ["Feature1", "Feature2"]
    elif dataset_name == "Iris":
        data = load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names
    elif dataset_name == "Wine":
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
    elif dataset_name == "Upload Your Own" and user_file is not None:
        df = pd.read_csv(user_file)
        return df, None, None
    else:
        X, y, feature_names = None, None, []
    return X, y, feature_names

if dataset_name == "Upload Your Own":
    if user_file is None:
        st.warning("Please upload a CSV file.")
        st.stop()

    df, _, _ = get_dataset(dataset_name, user_file)
    st.write("📄 Uploaded Data Preview", df.head())

    target_column = st.selectbox("🎯 Select Target Column", df.columns)
    feature_columns = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect("📊 Select 2 Features for Plotting", feature_columns, default=feature_columns[:2])

    show_plot = False
    if len(selected_features) < 1:
        st.warning("Please select at least one feature.")
        st.stop()
    elif len(selected_features) == 1:
        st.info("Using one feature only. Decision boundary will not be shown.")
        show_plot = False
    elif len(selected_features) == 2:
        show_plot = True

    X_df = df[selected_features].copy()
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col])
    X = X_df.values
    y = df[target_column].values
    feature_names = selected_features
else:
    X, y, feature_names = get_dataset(dataset_name, user_file)
    selected_features = st.multiselect("📊 Select 2 Features for Plotting", feature_names, default=feature_names[:2])
    if len(selected_features) < 2:
        st.warning("Please select 2 features for plotting.")
        st.stop()
    feature_indices = [feature_names.index(f) for f in selected_features]
    X = X[:, feature_indices]
    feature_names = selected_features
    show_plot = True

scale_data = model_name in ["Logistic Regression", "K-Nearest Neighbors"]
# Automatically scale data if algorithm requires it
if scale_data:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if dataset_name == "Upload Your Own":
        st.info("📏 Your uploaded features are being automatically scaled for better accuracy with this algorithm.")
    with st.expander("🔍 Scaler Preview (StandardScaler)"):
        st.write(pd.DataFrame(X, columns=feature_names).head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Ensure model is trained and tested on same feature dimension
if X_train.shape[1] != X_test.shape[1]:
    st.error(f"Mismatch in train/test dimensions: {X_train.shape[1]} vs {X_test.shape[1]}")
    st.stop()

if model_name == "Decision Tree":
    if X.shape[1] == 1 and model_name in ["Decision Tree", "Random Forest", "K-Nearest Neighbors", "Logistic Regression"]:
        st.warning("Your dataset only has one feature. The model will still run, but ensure your data is meaningful for prediction.")
    model = DecisionTreeClassifier(criterion=criterion,
                                   splitter=splitter,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   random_state=42)
elif model_name == "Random Forest":
    if X.shape[1] == 1 and model_name in ["Decision Tree", "Random Forest", "K-Nearest Neighbors", "Logistic Regression"]:
        st.warning("Your dataset only has one feature. The model will still run, but ensure your data is meaningful for prediction.")
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
elif model_name == "Logistic Regression":
    if X.shape[1] == 1 and model_name in ["Decision Tree", "Random Forest", "K-Nearest Neighbors", "Logistic Regression"]:
        st.warning("Your dataset only has one feature. The model will still run, but ensure your data is meaningful for prediction.")
    model = LogisticRegression(max_iter=1000)
elif model_name == "K-Nearest Neighbors":
    if X.shape[1] == 1 and model_name in ["Decision Tree", "Random Forest", "K-Nearest Neighbors", "Logistic Regression"]:
        st.warning("Your dataset only has one feature. The model will still run, but ensure your data is meaningful for prediction.")
    k = st.sidebar.slider("n_neighbors", 1, 20, 5)
    model = KNeighborsClassifier(n_neighbors=k)

@st.cache_resource
def train_model(_model, _X_train, _y_train):
    _model.fit(_X_train, _y_train)
    return _model

with st.spinner("Training model..."):
    model = train_model(model, X_train, y_train)

# Ensure model expects the same number of features as provided
if hasattr(model, 'n_features_in_') and X_test.shape[1] != model.n_features_in_:
    st.error(f"Model expects {model.n_features_in_} features, but got {X_test.shape[1]}.")
    st.stop()

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.markdown(f"### ✅ Accuracy Score: `{acc:.2f}`")

def plot_decision_boundary(clf, X, y, model_name=""):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .01  

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k', ax=ax)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{model_name} Classification Boundary")
    st.pyplot(fig)

if 'show_plot' in locals() and show_plot and st.checkbox("Show Decision Boundary", value=True):
    plot_decision_boundary(model, X_test, y_test, model_name)

if st.checkbox("Show Confusion Matrix and Classification Report", value=True):
    st.markdown("### 📊 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("### 🔍 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax_cm, cmap='Blues')
    st.pyplot(fig_cm)


if model_name == "Decision Tree" and len(feature_names) > 0:
    st.markdown("### 🌲 Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=[str(cls) for cls in np.unique(y)])
    st.pyplot(fig)
