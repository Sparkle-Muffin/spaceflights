"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["model_input_table", "params:model_options"],
            outputs=["X_train", "X_test", "y_train", "y_test"]
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train"],
            outputs="regressor"
        ),
        node(
            func=evaluate_model,
            inputs=["regressor", "X_test", "y_test"],
            outputs=None
        )
    ])
