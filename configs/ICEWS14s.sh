#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/ICEWS14s/"
vocab_dir="datasets/data_preprocessed/ICEWS14s/vocab"
total_iterations=100
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.05
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
train_tim_embeddings=1
base_output_dir="output/ICEWS14s"
load_model=0
model_load_dir=""
nell_evaluation=1
#load_model=1
#model_load_dir="saved_models/ICEWS14s/model.ckpt"
