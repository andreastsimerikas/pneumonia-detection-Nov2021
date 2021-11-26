from pandas.core.frame import DataFrame


train_path = 'pneumonia/train_images/images'
train_batches = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_path,
    x_col="file_name",
    y_col="class_id",
    batch_size=b_size,
    subset='training',
    shuffle=True,
    seed=42,
    class_mode="categorical",
    target_size=(224,224)
)

valid_batches = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_path,
    x_col="file_name",
    y_col="class_id",
    subset="validation",
    batch_size=b_size,
    shuffle=True,
    seed=42,
    class_mode="categorical",
    target_size=(224,224)
)