import xgboost as xgb
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import Categorizer
import coiled
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

# Create a local Dask Cluster
cluster = LocalCluster()  # all the power of our local machine

# or a Coiled Cloud cluster
cluster = coiled.Cluster(
    n_workers=4,
    worker_cpu=4,
    worker_memory='24GiB',  # As much as we need
    software='michal-mucha/xgboost-on-coiled'  # Public docker image
)

# Connect to the cluster
client = Client(cluster)

# Load the example dataset sample - specify columns
columns = [
    "interest_rate", "loan_age", "num_borrowers",
    "borrower_credit_score", "num_units"
]

categorical = [
    "orig_channel", "occupancy_status", "property_state",
    "first_home_buyer", "loan_purpose", "property_type",
    "zip", "relocation_mortgage_indicator", "delinquency_12"
]

# Download data from S3
mortgage_data = dd.read_parquet(
    "s3://coiled-data/mortgage-2000.parq/*",
    compression="gzip",
    columns=columns + categorical,
    storage_options={"anon": True}
)

# Cache the data on Cluster workers
mortgage_data = mortgage_data.persist()

# Cast categorical columns to the correct type

ce = Categorizer(columns=categorical)
mortgage_data = ce.fit_transform(mortgage_data)
for col in categorical:
    mortgage_data[col] = mortgage_data[col].cat.codes


# Create the train-test split

X, y = mortgage_data.iloc[:, :-1], mortgage_data["delinquency_12"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True, random_state=2
)

# Create the XGBoost DMatrix

dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)

# Set parameters
params = {
    "max_depth": 8,
    "max_leaves": 2 ** 8,
    "gamma": 0.1,
    "eta": 0.1,
    "min_child_weight": 30,
    "objective": "binary:logistic",
    "grow_policy": "lossguide"
}


# train the model
# % % time
output = xgb.dask.train(
    client, params, dtrain, num_boost_round=5,
    evals=[(dtrain, 'train')]
)

booster = output['booster']  # booster is the trained model
history = output['history']  # A dictionary containing evaluation

# Set down the cluster
client.close()
