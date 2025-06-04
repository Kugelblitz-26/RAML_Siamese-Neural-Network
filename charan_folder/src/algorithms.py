import networkx as nx

def build_dependency_graph(messages):
    G = nx.DiGraph()
    for msg in messages:
        G.add_edge(msg["sender"], msg["receiver"])
    return G

def schedule_single_node(application_data, policy):
    tasks = application_data["tasks"]
    messages = application_data.get("messages", [])

    G = build_dependency_graph(messages)
    task_map = {t["id"]: t for t in tasks}
    scheduled = {}
    schedule = []
    missed_deadlines = []

    available = [t for t in G.nodes if G.in_degree(t) == 0]
    current_time = 0

    while available:
        # Sort based on policy
        if policy == "edf":
            available.sort(key=lambda t: task_map[t]["deadline"])
        elif policy == "ldf":
            available.sort(key=lambda t: -task_map[t]["deadline"])
        elif policy == "ll":
            available.sort(key=lambda t: task_map[t]["deadline"] - task_map[t]["wcet"])

        task_id = available.pop(0)
        task = task_map[task_id]

        preds = list(G.predecessors(task_id))
        if any(p in missed_deadlines for p in preds):
            missed_deadlines.append(task_id)
            G.remove_node(task_id)
            continue

        ready_time = max(scheduled[p]["end_time"] for p in preds) if preds else 0
        start_time = max(current_time, ready_time)
        end_time = start_time + task["wcet"]

        if end_time > task["deadline"]:
            missed_deadlines.append(task_id)
            G.remove_node(task_id)
            continue

        current_time = end_time
        entry = {
            "task_id": task_id,
            "node_id": 0,
            "start_time": start_time,
            "end_time": end_time,
            "deadline": task["deadline"]
        }

        scheduled[task_id] = entry
        schedule.append(entry)
        G.remove_node(task_id)

        # Add newly available tasks
        for succ in list(G.nodes):
            if (
                succ not in scheduled
                and succ not in missed_deadlines
                and all(p in scheduled for p in G.predecessors(succ))
                and succ not in available
            ):
                available.append(succ)

    return {
        "schedule": schedule,
        "missed_deadlines": missed_deadlines,
        "name": f"{policy.upper()} Singlecore (forward scheduling)"
    }



def schedule_multi_node(application_data, platform_data, policy):
    tasks = application_data["tasks"]
    messages = application_data.get("messages", [])
    num_nodes = len(platform_data.get("nodes", [{"id": 0}]))

    G = build_dependency_graph(messages)
    task_map = {t["id"]: t for t in tasks}
    scheduled = {}
    schedule = []
    node_time = [0] * num_nodes

    # Initially schedulable tasks (leaf nodes)
    available = [t for t in G.nodes if G.in_degree(t) == 0]

    while available:
        # Sort according to policy
        if policy == "edf":
            available.sort(key=lambda t: task_map[t]["deadline"])
        elif policy == "ldf":
            available.sort(key=lambda t: -task_map[t]["deadline"])
        elif policy == "ll":
            def laxity(t):
                return task_map[t]["deadline"] - task_map[t]["wcet"]
            available.sort(key=laxity)

        task_id = available.pop(0)
        task = task_map[task_id]

        preds = list(G.predecessors(task_id))
        ready_time = max(scheduled[p]["end_time"] for p in preds) if preds else 0

        # Choose earliest available node
        earliest_start = float("inf")
        best_node = 0
        for node_id in range(num_nodes):
            start = max(node_time[node_id], ready_time)
            if start < earliest_start:
                earliest_start = start
                best_node = node_id

        start_time = earliest_start
        end_time = start_time + task["wcet"]
        node_time[best_node] = end_time

        entry = {
            "task_id": task_id,
            "node_id": best_node,
            "start_time": start_time,
            "end_time": end_time,
            "deadline": task["deadline"]
        }

        scheduled[task_id] = entry
        schedule.append(entry)
        G.remove_node(task_id)

        # Check which successors are now ready (all their preds are scheduled)
        for succ in list(G.nodes):
            if succ not in scheduled and all(p in scheduled for p in G.predecessors(succ)) and succ not in available:
                available.append(succ)

    missed_deadlines = [t["task_id"] for t in schedule if t["end_time"] > t["deadline"]]
    return {
        "schedule": schedule,
        "missed_deadlines": missed_deadlines,
        "name": f"{policy.upper()} Multinode (forward scheduling)"
    }



def edf_single_node(application_data):
    return schedule_single_node(application_data, "edf")

def ldf_single_node(application_data):
    return schedule_single_node(application_data, "ldf")


def edf_multinode_no_delay(application_data, platform_data):
    return schedule_multi_node(application_data, platform_data, "edf")

def ldf_multinode_no_delay(application_data, platform_data):
    return schedule_multi_node(application_data, platform_data, "ldf")

def ll_multinode_no_delay(application_data, platform_data):
    return schedule_multi_node(application_data, platform_data, "ll")