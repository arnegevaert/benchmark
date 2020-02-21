from sklearn.preprocessing import LabelEncoder


def recognize_features_type(df):
    integer_features = list(df.select_dtypes(include=['int64']).columns)
    double_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }
    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'

    return type_features, features_type


def set_discrete_continuous(features, type_features, class_name, discrete=None, continuous=None):
    if discrete is None and continuous is None:
        discrete = type_features['string']
        continuous = type_features['integer'] + type_features['double']

    if discrete is None and continuous is not None:
        discrete = [f for f in features if f not in continuous]
        continuous = list(set(continuous + type_features['integer'] + type_features['double']))

    if continuous is None and discrete is not None:
        continuous = [f for f in features if
                      f not in discrete and (f in type_features['integer'] or f in type_features['double'])]
        discrete = list(set(discrete + type_features['string']))

    discrete = [f for f in discrete if f != class_name] + [class_name]
    continuous = [f for f in continuous if f != class_name]
    return discrete, continuous


def label_encode(df, columns, label_encoder=None):
    df_le = df.copy(deep=True)
    new_le = label_encoder is None
    label_encoder = dict() if new_le else label_encoder
    for col in columns:
        if new_le:
            le = LabelEncoder()
            df_le[col] = le.fit_transform(df_le[col])
            label_encoder[col] = le
        else:
            le = label_encoder[col]
            df_le[col] = le.transform(df_le[col])
    return df_le, label_encoder
