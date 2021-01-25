"""This script processes the dataset from v1 to v2."""

import os
import yaml
import shutil


root_path = "/home/hao.zhang/project/pycharm/autodist/autodist/simulator/dataset/autosync_release_data"
target_path = "/home/hao.zhang/project/pycharm/autodist/autodist/simulator/dataset/autosync_release_data_v2"

if not os.path.exists(target_path):
    os.mkdir(target_path)

for model in os.listdir(root_path):
    model_folder = os.path.join(root_path, model)

    if model == "densenets169":
        target_model_folder = os.path.join(target_path, "densenet169")
    elif model == "densenets201":
        target_model_folder = os.path.join(target_path, "densenet201")
    else:
        target_model_folder = os.path.join(target_path, model)
    if not os.path.exists(target_model_folder):
        os.mkdir(target_model_folder)
    for cluster in os.listdir(model_folder):
        cluster_folder = os.path.join(model_folder, cluster)
        if cluster == "cluster1":
            target_cluster_folder = os.path.join(target_model_folder, "cluster-B")
        elif cluster == "cluster2":
            target_cluster_folder = os.path.join(target_model_folder, "cluster-A")

        if not os.path.exists(target_cluster_folder):
            os.mkdir(target_cluster_folder)

        # create the runtime, resource_spec, and strategy folders in the target dir
        target_runtime_folder = os.path.join(target_cluster_folder, "runtime")
        if not os.path.exists(target_runtime_folder):
            os.mkdir(target_runtime_folder)
        target_resource_spec_folder = os.path.join(target_cluster_folder, "resource_spec")
        if not os.path.exists(target_resource_spec_folder):
            os.mkdir(target_resource_spec_folder)
        target_strategy_folder = os.path.join(target_cluster_folder, "strategy")
        if not os.path.exists(target_strategy_folder):
            os.mkdir(target_strategy_folder)
        for experiment in os.listdir(cluster_folder):
            experiment_folder = os.path.join(cluster_folder, experiment)
            print("Processing folder {}...".format(experiment_folder))
            # find the resource spec.
            resource_spec_file_path = None
            for folder in os.listdir(experiment_folder):
                if not folder.endswith(".yml"):
                    continue
                resource_spec_file_path = os.path.join(experiment_folder, folder)
            if not os.path.exists(resource_spec_file_path):
                raise RuntimeError()
            for folder in os.listdir(experiment_folder):
                if folder == "runtimes":
                    runtime_folder = os.path.join(experiment_folder, folder)
                    for runtime_file in os.listdir(runtime_folder):
                        runtime_file_path = os.path.join(runtime_folder, runtime_file)

                        # now copy the strategy file
                        source_strategy_file = os.path.join(experiment_folder, "strategies", runtime_file)
                        if not os.path.exists(source_strategy_file):
                            print("Exception: file {} does not have a strategy".format(runtime_file))
                            continue

                        # copy them
                        with open(runtime_file_path) as file:
                            runtime_dict = yaml.load(file, Loader=yaml.FullLoader)
                            new_runtime_dict = {}
                            new_runtime_dict["average"] = runtime_dict["average"]
                            new_runtime_dict["runtime"] = runtime_dict["runtime"]
                        target_runtime_file = os.path.join(target_runtime_folder, runtime_file)
                        with open(target_runtime_file, "w") as file:
                            yaml.dump(new_runtime_dict, file)

                        # now copy the resource spec file
                        target_resource_spec_file = os.path.join(target_resource_spec_folder, runtime_file)
                        shutil.copy(resource_spec_file_path, target_resource_spec_file)
                        target_strategy_file = os.path.join(target_strategy_folder, runtime_file)
                        shutil.copy(source_strategy_file, target_strategy_file)
