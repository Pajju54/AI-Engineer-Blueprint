from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from features.features import ColumnSelector, CustomLogTransformer


def build_preprocessing_pipeline():
    numeric_cols = ['Age','SibSp','Parch','Fare']
    categorical_cols = ['Pclass','Sex','Embarked']
    log_transform_cols = ['Fare']

    numeric_pipeline = Pipeline([
        ('select', ColumnSelector(columns=numeric_cols)),
        ('impute', SimpleImputer(strategy='median')),
        ('log', CustomLogTransformer(columns=log_transform_cols)),
        ('scale', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('select', ColumnSelector(columns=categorical_cols)),
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))    
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    return full_pipeline