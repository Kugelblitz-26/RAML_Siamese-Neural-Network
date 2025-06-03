def _dummy_scheduler(application_data, algorithm_name):
    schedule = []
    current_time = 0

    for task in application_data["tasks"]:
        task_id = task["id"]
        deadline = task["deadline"]
        wcet = task["wcet"]
        node_id = 0  # assign everything to node 0 for now

        start_time = current_time
        end_time = start_time + wcet

        schedule.append({
            "task_id": task_id,
            "node_id": node_id,
            "start_time": start_time,
            "end_time": end_time,
            "deadline": deadline
        })

        current_time = end_time

    return {
        "schedule": schedule,
        "name": algorithm_name,
        "missed_deadlines": []
    }

def ldf_single_node(application_data):
    return _dummy_scheduler(application_data, "LDF Single-node")

def edf_single_node(application_data):
    return _dummy_scheduler(application_data, "EDF Single-node")

def rms_singlecore(application_data):
    return _dummy_scheduler(application_data, "RMS Single-node")

def ll_singlecore(application_data):
    return _dummy_scheduler(application_data, "LL Single-node")

def ldf_multinode_no_delay(application_data, platform_data):
    return _dummy_scheduler(application_data, "LDF Multinode(without delay)")

def edf_multinode_no_delay(application_data, platform_data):
    return _dummy_scheduler(application_data, "EDF Multinode(without delay)")

def ll_multinode_no_delay(application_data, platform_data):
    return _dummy_scheduler(application_data, "LL(without delay)")
