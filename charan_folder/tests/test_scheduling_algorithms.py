import pytest
import os
import json
import jsonschema
from src.algorithms import ldf_single_node, edf_single_node, edf_multinode_no_delay, ldf_multinode_no_delay, ll_multinode_no_delay

# Adjust path to include the 'src' directory for importing algorithms
script_dir = os.path.dirname(__file__)
input_models_dir = os.path.join(script_dir, "input_models")
output_schema_file = os.path.join(script_dir, "..", "src", "output_schema.json")

with open(output_schema_file) as f:
    output_schema = json.load(f)

input_files = os.listdir(input_models_dir)

algorithms = [ldf_single_node, edf_single_node, edf_multinode_no_delay, ldf_multinode_no_delay, ll_multinode_no_delay]
algorithms_multinode_no_delay = [edf_multinode_no_delay, ldf_multinode_no_delay, ll_multinode_no_delay]

# Creating a product of filenames and algorithms for detailed parameterization
test_cases = [(input_file, algo) for input_file in input_files for algo in algorithms]
test_cases_multinode_no_delay = [(input_file, algo)
                                 for input_file in input_files for algo in algorithms_multinode_no_delay]


def load_and_schedule(filename, algo):
    """Load the input model and run the scheduling algorithm"""
    model_path = os.path.join(input_models_dir, filename)
    with open(model_path) as f:
        model_data = json.load(f)

    application_model = model_data["application"]
    platform_model = model_data["platform"]

    if algo in [ldf_single_node, edf_single_node]:
        result = algo(application_model)

    elif algo in [edf_multinode_no_delay, ldf_multinode_no_delay, ll_multinode_no_delay]:
        result = algo(application_model, platform_model)

    return result, application_model, platform_model


@pytest.mark.parametrize("filename,algorithm", test_cases)
def test_output_schema(filename, algorithm):
    """Test that the generated schedule adheres to the schema """

    result, application_model, platform_model = load_and_schedule(filename, algorithm)
    try:
        jsonschema.validate(instance=result, schema=output_schema)
        assert True
    except jsonschema.exceptions.ValidationError as err:
        assert False, f'Output does not match the output schema for {result["name"]}'


@pytest.mark.parametrize("filename, algorithm", test_cases)
def test_task_duration(filename, algorithm):
    """Test that each task completes within its estimated duration."""
    result, app_model, platform_model = load_and_schedule(filename, algorithm)
    for task in result["schedule"]:
        start_time = task["start_time"]
        end_time = task["end_time"]
        task_id = task["task_id"]
        wcet = next(
            (t["wcet"] for t in app_model["tasks"] if t["id"] == task_id), None
        )
        if wcet is None:
            raise ValueError(f'WCET not found for task {task_id}, in {result["name"]}')
        assert end_time == start_time + wcet, f'Incorrect task duration calculation in {result["name"]}'


@pytest.mark.parametrize("filename, algorithm", test_cases)
def test_task_deadline(filename, algorithm):
    """Test that each task respects its deadline."""
    result, app_model, platform_model = load_and_schedule(filename, algorithm)
    for task in result["schedule"]:
        end_time = task["end_time"]
        task_id = task["task_id"]
        deadline = next(
            (t["deadline"] for t in app_model["tasks"] if t["id"] == task_id), None
        )
        assert end_time <= deadline, f'Task exceeds deadline in {result["name"]}'


@pytest.mark.parametrize("filename, algorithm", test_cases)
def test_task_dependencies(filename, algorithm):
    """Test that each task respects the completion times of its predecessors."""
    result, app_model, platform_model = load_and_schedule(filename, algorithm)
    for task in result["schedule"]:
        start_time = task["start_time"]
        task_id = task["task_id"]
        predecessors = [
            msg["sender"]
            for msg in app_model["messages"]
            if msg["receiver"] == task_id
        ]

        predecessors_end_times = [
            t["end_time"]
            for t in result["schedule"]
            if t["task_id"] in predecessors
        ]

        assert start_time >= max(
            predecessors_end_times, default=0
        ), f'Task starts before predecessor ends in {result["name"]}'


@pytest.mark.parametrize("filename, algorithm", test_cases)
def test_no_overlapping_tasks(filename, algorithm):
    """Test that no tasks overlap in their execution time on the same node."""
    result, application_model, platform_model = load_and_schedule(filename, algorithm)
    tasks_by_node = {}
    for task in result["schedule"]:
        node_id = task["node_id"]
        if node_id not in tasks_by_node:
            tasks_by_node[node_id] = []
        tasks_by_node[node_id].append(task)

    for node_id, tasks in tasks_by_node.items():
        tasks.sort(key=lambda t: t["start_time"])
        for i in range(1, len(tasks)):
            prev_task = tasks[i - 1]
            current_task = tasks[i]
            assert prev_task["end_time"] <= current_task["start_time"], \
                f'Tasks overlap on node {node_id} in {result["name"]}'


@pytest.mark.parametrize("filename, algorithm", test_cases)
def test_algorithm_correctness(filename, algorithm):

    result, app_model, platform_model = load_and_schedule(filename, algorithm)

    expected_results_dir = os.path.join(script_dir, "expected_results")
    expected_output_file = os.path.join(expected_results_dir, filename)

    with open(expected_output_file) as f:
        expected_output = json.load(f)
    actual_schedule = {task["task_id"]: task["start_time"] for task in result["schedule"]}
    print(f'Actual schedule: {actual_schedule}')

    expected_schedule = {task["task_id"]: task["start_time"] for task in expected_output[result["name"]]["schedule"]}
    print(f'Expected schedule: {expected_schedule}')

    for task_id, actual_start_time in actual_schedule.items():
        expected_start_time = expected_schedule.get(task_id)
        assert actual_start_time == expected_start_time, \
            f"Test_file {filename}, task {task_id}: start_time {actual_start_time}, Expected start_time {expected_start_time}"


@pytest.mark.parametrize("filename, algorithm", test_cases)
def test_missed_deadline(filename, algorithm):
    result, app_model, platform_model = load_and_schedule(filename, algorithm)
    # print(f'results_missed_deadline:{result.get("missed_deadlines", [])}')

    expected_results_dir = os.path.join(script_dir, "expected_results")
    expected_output_file = os.path.join(expected_results_dir, filename)

    with open(expected_output_file) as f:
        expected_output = json.load(f)
    # print(f'expected output:{expected_output[result["name"]]} filename:{filename}')

    actual_missed_deadline = result.get("missed_deadlines", [])
    # print(f'Actual missed deadline : {actual_missed_deadline}')

    expected_missed_deadline = expected_output[result["name"]].get("missed_deadlines", [])
    # print(f'Expected missed deadline: {expected_missed_deadline}')

    assert actual_missed_deadline == expected_missed_deadline,  f"missed deadlines do not match for {result['name']}"

# @pytest.mark.parametrize("filename, algorithm", test_cases_multinode)
# def test_dependency_multinode(filename, algorithm):
#     """Test that task dependencies are respected across nodes in multi-node scenarios.
#     This test is only applicable to multi-node scheduling algorithms with communication delays."""
#     result, application_model, platform_model = load_and_schedule(filename, algorithm)
#
#     tasks = application_model["tasks"]
#     messages = application_model["messages"]
#
#     application_graph = nx.DiGraph()
#
#     for task in tasks:
#         application_graph.add_node(task["id"], task_data=task)
#
#     for message in messages:
#         sender = message["sender"]
#         receiver = message["receiver"]
#         application_graph.add_edge(sender, receiver, message_data=message)
#
#     platform_graph = nx.Graph()
#
#     for node in platform_model["nodes"]:
#         platform_graph.add_node(node["id"], type=node["type"])
#
#     for link in platform_model["links"]:
#         platform_graph.add_edge(link["start_node"], link["end_node"], weight=link["link_delay"])
#
#     shortest_paths = dict(nx.all_pairs_dijkstra_path_length(platform_graph, weight='weight'))
#
#     for task in result["schedule"]:
#         start_time = task["start_time"]
#         task_id = task["task_id"]
#         current_node = task["node_id"]
#         predecessors = [
#             msg["sender"]
#             for msg in application_model["messages"]
#             if msg["receiver"] == task_id
#         ]
#
#         for pred_id in predecessors:
#             pred_task = next(t for t in result["schedule"] if t["task_id"] == pred_id)
#             pred_node = pred_task["node_id"]
#             pred_end_time = pred_task["end_time"]
#
#             communication_delay = shortest_paths[pred_node][current_node]
#             earliest_start_time = pred_end_time + communication_delay
#
#             assert start_time >= earliest_start_time, \
#                 f'Task {task_id} starts before predecessor' +\
#                 f'{pred_id} ends considering communication delay in {result["name"]}'
