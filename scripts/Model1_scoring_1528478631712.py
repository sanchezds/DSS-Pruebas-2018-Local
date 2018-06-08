# Copyright 2017, 2018 IBM. IPLA licensed Sample Materials.


from mlpipelinepy.mlpipeline import MLPipelineModel
from mlpipelinepy import SparkDataSources
from pyspark.sql import SparkSession
import os
import json
import pandas
import pylru #python 2.6 compatible LRU cache
import pyspark

#globals
global model
model = None
global spark
spark = None
model_cache = pylru.lrucache(33)


def init():
    global model
    global spark

    global model
    global serialization_method

    model_name = "Model1"
    version = "1"
    project_name = os.environ.get("DSX_PROJECT_NAME")
    user_id = os.environ.get("DSX_USER_ID", "990")
    project_path = "/user-home/" + user_id + "/DSX_Projects/" + project_name
    model_parent_path = project_path + "/models/" + model_name + "/"
    metadata_path = model_parent_path + "metadata.json"

    # fetch info from metadata.json
    with open(metadata_path) as data_file:
        meta_data = json.load(data_file)

    # if latest version, find latest version from  metadata.json
    if (version == "latest"):
        version = meta_data.get("latestModelVersion")

    # prepare model path using model name and version
    model_path = model_parent_path + str(version) + "/model"

    serialization_method = "spark"

    # create spark context
    sc = pyspark.SparkContext("local[*]")
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # load model
    model_id = model_name + str(version)
    if model_id not in model_cache:
        model_cache[model_id] = MLPipelineModel.load(model_path)
    model = model_cache[model_id]

def score(args):
    global model
    global spark

    input_json_str = args.get("input_json_str")

    pandaDF = pandas.read_json(input_json_str)
    df = spark.createDataFrame(pandaDF)

    out = model.transform(SparkDataSources({ 'nodeADP': df }))
    result = out[0].data_frame.collect()[0]

    return json.dumps({
        "classes": result.nodeADP_classes if 'nodeADP_classes' in result else [],
        "predictions": result.prediction if isinstance(result.prediction, list) else [result.prediction],
        "probabilities": result.probability.tolist() if 'probability' in result else []
    })



def test_score(args):
    """Call this method to score in development."""
    init()
    return score(args)