import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gc

from tensorflow.keras import layers, optimizers, losses, Model
from sklearn import preprocessing, model_selection

from func import load_data_2_create_graph, get_graph_data, node_generator, get_x_in_prediction, get_model


# Load data and create graph via Networkx
G_m1 = load_data_2_create_graph('input_5000/m1.csv')
G_m2 = load_data_2_create_graph('input_5000/m2.csv')
G_m3 = load_data_2_create_graph('input_5000/m3.csv')
G_m4 = load_data_2_create_graph('input_5000/m4.csv')

# Load note features X and labels
X = pd.read_csv('input_5000/app_api.csv', header=0, index_col='Unnamed: 0')
labels = X.pop('Label')
labels_sampled = labels.sample(frac=0.8, replace=False, random_state=42)

# Create StellarGraph Data Object from above graphs (from_networkx function)
# m1
graph_full_m1, graph_sampled_m1 = get_graph_data(G_m1, X, labels_sampled)
del G_m1
gc.collect()

# m2
graph_full_m2, graph_sampled_m2 = get_graph_data(G_m2, X, labels_sampled)
del G_m2
gc.collect()

# m3
graph_full_m3, graph_sampled_m3 = get_graph_data(G_m3, X, labels_sampled)
del G_m3
gc.collect()

# m4
graph_full_m4, graph_sampled_m4 = get_graph_data(G_m4, X, labels_sampled)
del G_m4
gc.collect()

# Split data
train_labels, test_labels = model_selection.train_test_split(
    labels_sampled, train_size=0.6, test_size=None, stratify=labels_sampled, random_state=42,
)
val_labels, test_labels = model_selection.train_test_split(
    test_labels, train_size=0.6, test_size=None, stratify=test_labels, random_state=42,
)

# Encode label
target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_labels)
val_targets = target_encoding.transform(val_labels)
test_targets = target_encoding.transform(test_labels)

# Define batch_size, num_samples
batch_size = 100
num_samples = [10, 10]
layer_sizes = [64, 32]

# Node Generator
# m1
generator_m1, train_gen_m1, val_gen_m1 = node_generator(graph_sampled_m1, train_labels, train_targets, val_labels,
                                                        val_targets, batch_size, num_samples)

# m2
generator_m2, train_gen_m2, val_gen_m2 = node_generator(graph_sampled_m2, train_labels, train_targets, val_labels,
                                                        val_targets, batch_size, num_samples)

# m3
generator_m3, train_gen_m3, val_gen_m3 = node_generator(graph_sampled_m3, train_labels, train_targets, val_labels,
                                                        val_targets, batch_size, num_samples)

# m4
generator_m4, train_gen_m4, val_gen_m4 = node_generator(graph_sampled_m4, train_labels, train_targets, val_labels,
                                                        val_targets, batch_size, num_samples)

for dropout in [0.1]:
    for lr in [0.003]:
        # m1
        x_inp_m1, prediction_m1 = get_x_in_prediction(generator_m1, train_targets, layer_sizes, dropout)

        # m2
        x_inp_m2, prediction_m2 = get_x_in_prediction(generator_m2, train_targets, layer_sizes, dropout)

        # m3
        x_inp_m3, prediction_m3 = get_x_in_prediction(generator_m3, train_targets, layer_sizes, dropout)

        # m4
        x_inp_m4, prediction_m4 = get_x_in_prediction(generator_m3, train_targets, layer_sizes, dropout)

        # Model
        # m1
        model_m1 = get_model(x_inp_m1, prediction_m1, lr)

        # m2
        model_m2 = get_model(x_inp_m2, prediction_m2, lr)

        # m3
        model_m3 = get_model(x_inp_m3, prediction_m3, lr)

        # m4
        model_m4 = get_model(x_inp_m4, prediction_m4, lr)

        # Combine 4 Model
        combined = layers.concatenate([model_m1.output, model_m2.output, model_m3.output, model_m4.output])
        z = layers.Dense(units=train_targets.shape[1], activation="softmax")(combined)
        combined_model = Model(inputs=[x_inp_m1, x_inp_m2, x_inp_m3, x_inp_m4], outputs=z)
        combined_model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )

        # Input
        label = np.vstack([train_gen_m1[i][1] for i in range(len(train_gen_m1))])
        input_1 = np.vstack(np.array([train_gen_m1[i][0][0] for i in range(len(train_gen_m1))]))
        input_2 = np.vstack(np.array([train_gen_m1[i][0][1] for i in range(len(train_gen_m1))]))
        input_3 = np.vstack(np.array([train_gen_m1[i][0][2] for i in range(len(train_gen_m1))]))
        input_4 = np.vstack(np.array([train_gen_m2[i][0][0] for i in range(len(train_gen_m2))]))
        input_5 = np.vstack(np.array([train_gen_m2[i][0][1] for i in range(len(train_gen_m2))]))
        input_6 = np.vstack(np.array([train_gen_m2[i][0][2] for i in range(len(train_gen_m2))]))
        input_7 = np.vstack(np.array([train_gen_m3[i][0][0] for i in range(len(train_gen_m3))]))
        input_8 = np.vstack(np.array([train_gen_m3[i][0][1] for i in range(len(train_gen_m3))]))
        input_9 = np.vstack(np.array([train_gen_m3[i][0][2] for i in range(len(train_gen_m3))]))
        input_10 = np.vstack(np.array([train_gen_m4[i][0][0] for i in range(len(train_gen_m4))]))
        input_11 = np.vstack(np.array([train_gen_m4[i][0][1] for i in range(len(train_gen_m4))]))
        input_12 = np.vstack(np.array([train_gen_m4[i][0][2] for i in range(len(train_gen_m4))]))

        # Validation
        val_label = np.vstack([val_gen_m1[i][1] for i in range(len(val_gen_m1))])
        val_1 = np.vstack(np.array([val_gen_m1[i][0][0] for i in range(len(val_gen_m1))]))
        val_2 = np.vstack(np.array([val_gen_m1[i][0][1] for i in range(len(val_gen_m1))]))
        val_3 = np.vstack(np.array([val_gen_m1[i][0][2] for i in range(len(val_gen_m1))]))
        val_4 = np.vstack(np.array([val_gen_m2[i][0][0] for i in range(len(val_gen_m2))]))
        val_5 = np.vstack(np.array([val_gen_m2[i][0][1] for i in range(len(val_gen_m2))]))
        val_6 = np.vstack(np.array([val_gen_m2[i][0][2] for i in range(len(val_gen_m2))]))
        val_7 = np.vstack(np.array([val_gen_m3[i][0][0] for i in range(len(val_gen_m3))]))
        val_8 = np.vstack(np.array([val_gen_m3[i][0][1] for i in range(len(val_gen_m3))]))
        val_9 = np.vstack(np.array([val_gen_m3[i][0][2] for i in range(len(val_gen_m3))]))
        val_10 = np.vstack(np.array([val_gen_m4[i][0][0] for i in range(len(val_gen_m4))]))
        val_11 = np.vstack(np.array([val_gen_m4[i][0][1] for i in range(len(val_gen_m4))]))
        val_12 = np.vstack(np.array([val_gen_m4[i][0][2] for i in range(len(val_gen_m4))]))

        history_combined = combined_model.fit(
            x=[input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11,
               input_12], y=label,
            epochs=100,
            validation_data=([val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, val_9, val_10, val_11, val_12],
                             val_label),
            verbose=1,
            shuffle=True
        )

        del input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, \
            input_10, input_11, input_12, val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, \
            val_9, val_10, val_11, val_12, model_m1, model_m2, model_m3, model_m4, combined_model
        gc.collect()

        plt.plot(history_combined.history['acc'])
        plt.plot(history_combined.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.show()
        if not os.path.exists('output'):
            os.makedirs('output')
        plt.savefig(f'output/dropout_{dropout}_lr_{lr}.png')
        plt.clf()
