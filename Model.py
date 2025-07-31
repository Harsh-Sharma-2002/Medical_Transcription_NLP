from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def get_param_grid():
    return {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'bootstrap': [True, False]
    }


def get_grid_search_model(cv=3, scoring='accuracy', n_jobs=-1, verbose=1):
    model = RandomForestClassifier(random_state=42)
    param_grid = get_param_grid()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    return grid_search
