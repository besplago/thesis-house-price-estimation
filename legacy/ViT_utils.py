from sklearn.model_selection import train_test_split
from transformers import DefaultDataCollator
import tensorflow as tf
from tensorflow import keras

import numpy as np


# def augmentation(examples, feature_extractor):
#   data_augmentation = keras.Sequential(
#   [
#       layers.Resizing(feature_extractor.size, feature_extractor.size),
#       layers.Rescaling(1./255),
#       layers.RandomFlip("horizontal"),
#       layers.RandomRotation(factor=0.01),
#       layers.RandomZoom(
#           height_factor=0.05, width_factor=0.05
#       ),
#   ],
#   name="data_augmentation",
#   )
#   examples["pixel_values"] = [data_augmentation(image) for image in np.array(examples["image_floorplan_pp"])]
#   return examples

# def process(examples, feature_extractor, column_name): 
#   examples.update(feature_extractor(examples[column_name]))
  
# def id_label(bins): 
#   id2label = {i: label for i, label in enumerate(bins)}
#   label2id = {label: i for i, label in enumerate(bins)}
#   return id2label, label2id

# def convert_to_tf_dataset(data_set):
#   data_collator = DefaultDataCollator(return_tensors="tf")
#   return data_set.to_tf_dataset(columns=['pixel_values'],
#                                 label_cols=["labels"],
#                                 shuffle=True,
#                                 batch_size=32,
#                                 collate_fn=data_collator)

# def df_to_tf_dataset(images, labels):
#   tensors = [tf.convert_to_tensor(image) for image in images]       





#FOR HUGGINGFACE VIT
def augmentation(examples, feature_extractor): 
  data_augmentation = keras.Sequential(
  [
      tf.layers.Resizing(feature_extractor.size, feature_extractor.size),
      tf.layers.Rescaling(1./255),
      tf.layers.RandomFlip("horizontal"),
      tf.layers.RandomRotation(factor=0.01),
      tf.layers.RandomZoom(
          height_factor=0.05, width_factor=0.05
      ),
  ],
  name="data_augmentation",
  )
  examples["pixel_values"] = [data_augmentation(image) for image in np.array(examples["image_floorplan_pp"])]
  return examples

def process(img, feature_extractor): 
  return feature_extractor(img)


from datasets import load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
from transformers import TrainingArguments
from transformers import Trainer
import torch
from datasets import Dataset

def transform(example_batch, processor): 
  inputs = processor([x for x in np.array(example_batch['image_floorplan_pp'])], return_tensors='pt')
  inputs['labels'] = example_batch['labels']
  return inputs


def prepare_data(houses_df, image_column, label_column): 
  train_df, test_df = train_test_split(houses_df, test_size=0.2, random_state=42)
  train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42) 
  #take img and label
  train_ds = train_df[[image_column, label_column]]
  valid_ds = valid_df[[image_column, label_column]]
  test_ds = test_df[[image_column, label_column]]

  #Turn them into huggingface datasets 
  train_ds = Dataset.from_pandas(train_ds)
  valid_ds = Dataset.from_pandas(valid_ds)
  test_ds = Dataset.from_pandas(test_ds)  
  return train_ds, valid_ds, test_ds

metric = load_metric("accuracy")  
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def get_trainer(model, processor, output_dir, train_dataset, eval_dataset): 
  trainer_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
  )
  trainer = Trainer(
    model=model,
    args=trainer_args,
    data_collator=collate_fn,
    #compute_metrics=,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    
  )
  return trainer


def get_vit_model(model_name, num_labels, id_2_label, label_2_id): 
  model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=(num_labels),
    id2label=id_2_label,
    label2id=label_2_id
  )
  return model


def train_model(model, processor, train_ds, valid_ds, output_dir):
  trainer = get_trainer(model, processor, output_dir, train_ds, valid_ds)
  trainer.train()
  return trainer
  





############### USING KERAS ##################
def process(examples, feature_extractor, column_name): 
  #Turn examples[column_name] into (3,224,224) instead of (224,224,3)
  examples[column_name] = [np.moveaxis(image, -1, 0) for image in np.array(examples[column_name])]
  examples.update(feature_extractor(np.array(examples[column_name]), ))
  return examples

def augment(examples, feature_extractor): 
  data_augmentation = keras.Sequential(
  [
      tf.keras.layers.Resizing(feature_extractor.size, feature_extractor.size),
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.RandomFlip("horizontal"),
      tf.keras.layers.RandomRotation(factor=0.01),
      tf.keras.layers.RandomZoom(
          height_factor=0.05, width_factor=0.05
      ),
  ],
  name="data_augmentation",
  )
  examples["pixel_values"] = [data_augmentation(image) for image in np.array(examples["image_floorplan_pp"])]
  return examples

def prepare_data_keras(houses_df, image_column, label_column, feature_extractor): 
  data_collator = DefaultDataCollator(return_tensors="tf")
  houses_ds = Dataset.from_pandas(houses_df)
  processed_dataset = houses_ds.map(lambda x: process(x, feature_extractor, image_column), batched=True)
  processed_dataset = processed_dataset.shuffle().train_test_split(test_size=0.1)
  #Split it up again to gain validation set
  
  tf_train_dataset = processed_dataset['train'].to_tf_dataset(
    columns=[image_column],
    label_cols=[label_column],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator
  )
  tf_valid_dataset = processed_dataset['test'].to_tf_dataset(
    columns=[image_column],
    label_cols=[label_column],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator
  )
  return tf_train_dataset, tf_valid_dataset
  