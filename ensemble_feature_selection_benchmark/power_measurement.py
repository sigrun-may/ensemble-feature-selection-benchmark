# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Power consumption measurement for cluster nodes."""
from datetime import datetime

import time
from pprint import pprint

import requests
import xmltodict
from requests.auth import HTTPBasicAuth

from config import settings


def _get_actual_node_power_usage(node_power_usages_dict: dict):
    if settings["env"] == "cluster":
        # Making a get request
        api_response = requests.get(
            settings["monitoring_node"],
            auth=HTTPBasicAuth(
                settings["monitoring_node_user"], settings["monitoring_node_password"]
            ),
        )
        if api_response.status_code == 200:
            # all ok
            xml_data = api_response.text
            data_as_dict = xmltodict.parse(xml_data)
            for node in data_as_dict["nodeList"]["node"]:
                base_board_id_str = node["@baseBoardId"][
                    node["@baseBoardId"].find(start := "_BB_") + len(start)
                ]
                node_id = f"node{base_board_id_str}"
                if node_id in node_power_usages_dict.keys():
                    node_power_usages_dict[node_id] += float(
                        node["@actualNodePowerUsage"]
                    )
                    node_power_usages_dict[f"{node_id}_cpu"] += float(
                        node["@actualPowerUsage"]
                    )
                    node_power_usages_dict[f"{node_id}_peg"] += float(
                        node["@actualPEGPowerUsage"]
                    )
            node_power_usages_dict["number_of_measurements"] += 1
        else:
            node_power_usages_dict["error"] += 1
    else:
        # local testing
        for node in settings["nodes"]["baseBoardIds"]:
            node_id = f"node{str(node)}"
            if node_id in node_power_usages_dict.keys():
                node_power_usages_dict[node_id] += 1
                node_power_usages_dict[f"{node_id}_peg"] += 1
    return node_power_usages_dict


def _measure_power_usage(node_power_measurement_dict, event):
    if node_power_measurement_dict["workflow_element"] == "preprocessing":
        benchmark_interval = 1
    elif node_power_measurement_dict["workflow_element"] == "feature_selection":
        benchmark_interval = 60
    else:
        raise ValueError(
            "Workflow element can be 'preprocessing' or 'feature_selection'"
        )
    while True:
        time.sleep(benchmark_interval)
        node_power_measurement_dict = _get_actual_node_power_usage(
            node_power_measurement_dict
        )
        if event.is_set():
            break
    return node_power_measurement_dict


def _get_power_consumption_baseline(node_power_measurement_dict, seconds: int):
    counter = 0
    while counter < seconds:
        node_power_measurement_dict = _get_actual_node_power_usage(
            node_power_measurement_dict
        )
        time.sleep(1)
        counter += 1
        print(counter, node_power_measurement_dict)
    print(node_power_measurement_dict)

    for node_id, joule in node_power_measurement_dict.items():
        if node_id != "time":
            node_power_measurement_dict[node_id] = joule / seconds

    print(node_power_measurement_dict)
    return node_power_measurement_dict


def initialize_benchmark_dict(workflow_element):
    # initialize dict for power measurement
    benchmark_dict = {"error": 0, "number_of_measurements": 0}
    for node_id in settings["nodes"]["baseBoardIds"]:
        benchmark_dict[f"node{node_id}"] = 0
        benchmark_dict[f"node{node_id}_peg"] = 0
        benchmark_dict[f"node{node_id}_cpu"] = 0

    benchmark_dict["workflow_element"] = workflow_element
    benchmark_dict["time"] = datetime.now()
    assert type(benchmark_dict["time"]) == datetime
    return benchmark_dict
