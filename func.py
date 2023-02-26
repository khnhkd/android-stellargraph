import networkx as nx
import numpy as np
import pandas as pd
import gc
import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import layers, optimizers, losses, Model


def normalize_data(data):
    data = data.to_numpy()
    data = data / np.amax(data, axis=1)
    return np.tril(data)


def load_data_2_create_graph(data_path):
    adj_matr = pd.read_csv(data_path, index_col=None, header=None)
    adj_matr = normalize_data(adj_matr)
    G = nx.Graph(adj_matr)
    del adj_matr
    gc.collect()
    return G


def get_graph_data(graph, node_features, labels_sampled):
    graph_full = sg.StellarGraph.from_networkx(graph, node_features=node_features)
    graph_sampled = graph_full.subgraph(labels_sampled.index)
    return graph_full, graph_sampled


def node_generator(graph_sampled, train_labels, train_targets, val_labels, val_targets, batch_size, num_samples):
    generator = GraphSAGENodeGenerator(graph_sampled, batch_size, num_samples, weighted=True)
    train_gen = generator.flow(train_labels.index, train_targets, shuffle=True)
    val_gen = generator.flow(val_labels.index, val_targets)
    return generator, train_gen, val_gen


def get_x_in_prediction(generator, train_targets, layer_sizes, dropout):
    graphsage_model = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=dropout,
    )
    x_inp, x_out = graphsage_model.in_out_tensors()
    prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
    return x_inp, prediction


def get_model(x_inp, prediction, lr):
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    return model
