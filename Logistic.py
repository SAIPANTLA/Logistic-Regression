import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to generate U-shaped dataset
def generate_u_shape(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples, 2)
    y = (X[:, 1] > np.abs(X[:, 0] - 0.5)).astype(int)
    return X, y

# Function to generate Two Spirals dataset
def generate_two_spirals(n_points, noise=0.5):
    np.random.seed(0)
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return X, y

# Function to generate XOR dataset
def generate_xor(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples, 2) * 2 - 1
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

# Function to generate Overlap dataset
def generate_overlap(n_samples):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y

#Inno Logo
st.image(r"innomaticslogo.png")

#Title
st.markdown("<h1 style='text-align: center;'>Logistic Regression</h1>", unsafe_allow_html=True)


# Create a sidebar
st.sidebar.header("Hyperparameters")

# Choosing the Algorithms

# Add a select box to the sidebar
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Moons", "Blobs", "U-Shaped", "Circles", "XOR", "Two Spirals", "Overlap")
)

if dataset_name in ("Moons", "Circles", "Two Spirals", "Overlap", "XOR"):
    noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3, 0.1)
if dataset_name == "Circles":
    factor = st.sidebar.slider("Factor", 0.1, 0.99, 0.5, 0.1)
n_samples = 500

# Add sliders to the sidebar based on the dataset
if dataset_name == "Moons":
    X, y = make_moons(n_samples, noise=noise)

elif dataset_name == "Blobs":
    X, y = make_blobs(n_samples, cluster_std=st.sidebar.slider("Cluster Std", 0.0, 5.0, 1.0, 0.1))

elif dataset_name == "U-Shaped":
    X, y = generate_u_shape(n_samples)

elif dataset_name == "Circles":
    X, y = make_circles(n_samples, noise=noise, factor=factor)

elif dataset_name == "XOR":
    X, y = generate_xor(n_samples)

elif dataset_name == "Two Spirals":
    X, y = generate_two_spirals(n_samples, noise=noise)

elif dataset_name == "Overlap":
    X, y = generate_overlap(n_samples)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Logistic Regression parameters
penalty = st.sidebar.selectbox("Penalty", ('l1', 'l2', 'elasticnet', 'none'))

if penalty == 'none':
    penalty = None
    solver_options = ('lbfgs','newton-cg', 'newton-cholesky', 'sag', 'saga')
elif penalty == 'l1':
    solver_options = ('liblinear', 'saga')
elif penalty == 'l2':
    solver_options = ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga')
elif penalty == 'elasticnet':
    solver_options = ('saga',)

solver = st.sidebar.selectbox("Solver", solver_options)
fit_intercept = st.sidebar.radio("Fit Intercept", [True, False])
class_weight = st.sidebar.radio("Class Weight", ['balanced', None])

if solver_options == 'liblinear' or 'newton-cholesky':
    multi = ("auto", "ovr")
else:
    multi =("auto", "ovr", "multinomial")

multi_class = st.sidebar.selectbox("Multi-class", multi)

max_iter = st.sidebar.slider("Max Iterations", 50, 500, 100, 50)
warm_start = st.sidebar.radio("Warm Start", [True, False])
random_state=st.sidebar.number_input("Random State Number", min_value=0, max_value=1000, step=1)

# Initialize Logistic Regression model
params = {
    'penalty': penalty,
    'solver': solver,
    'fit_intercept': fit_intercept,
    'class_weight': class_weight,
    'multi_class': multi_class,
    'max_iter': max_iter,
    'warm_start': warm_start,
    'random_state': random_state
}

# Conditionally add 'dual' only if applicable
if solver == 'liblinear' and penalty == 'l2':
    params['dual'] = st.sidebar.radio("Dual", [True, False])
# Conditionally add 'l1_ratio' only if penalty is 'elasticnet'
if penalty == 'elasticnet':
    params['l1_ratio'] = st.sidebar.select_slider("L1 ratio", options=np.linspace(0.01, 0.99, num=99), value=0.1)


# Train Logistic Regression model
classifier = LogisticRegression(**params)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Plot the decision surface
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(6, 3))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"{dataset_name} Dataset\nAccuracy: {accuracy:.2f}")

# Display the plot in Streamlit
st.pyplot(plt)
