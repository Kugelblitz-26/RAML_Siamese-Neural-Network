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
    current_time = 0

    while len(scheduled) < len(tasks):
        ready_tasks = [
            tid for tid in G.nodes
            if tid not in scheduled and all(pred in scheduled for pred in G.predecessors(tid))
        ]
        if not ready_tasks:
            raise RuntimeError("Cyclic dependencies or no schedulable task.")

        if policy == "edf":
            ready_tasks.sort(key=lambda t: task_map[t]["deadline"])
        elif policy == "ldf":
            ready_tasks.sort(key=lambda t: -task_map[t]["deadline"])
        elif policy == "ll":
            ready_tasks.sort(key=lambda t: (task_map[t]["deadline"] - task_map[t]["wcet"] - current_time))

        task_id = ready_tasks[0]
        task = task_map[task_id]
        preds = list(G.predecessors(task_id))
        ready_time = max(scheduled[p]["end_time"] for p in preds) if preds else 0
        start_time = max(current_time, ready_time)
        end_time = start_time + task["wcet"]

        entry = {
            "task_id": task_id,
            "node_id": 0,
            "start_time": start_time,
            "end_time": end_time,
            "deadline": task["deadline"]
        }

        scheduled[task_id] = entry
        schedule.append(entry)
        current_time = end_time
        G.remove_node(task_id)

    missed_deadlines = [t["task_id"] for t in schedule if t["end_time"] > t["deadline"]]
    return {
        "schedule": schedule,
        "missed_deadlines": missed_deadlines,
        "name": f"{policy.upper()} Single-node"
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

    while len(scheduled) < len(tasks):
        ready_tasks = [
            tid for tid in G.nodes
            if tid not in scheduled and all(pred in scheduled for pred in G.predecessors(tid))
        ]
        if not ready_tasks:
            raise RuntimeError("Cyclic dependencies or no schedulable task.")

        if policy == "edf":
            ready_tasks.sort(key=lambda t: task_map[t]["deadline"])
        elif policy == "ldf":
            ready_tasks.sort(key=lambda t: -task_map[t]["deadline"])
        elif policy == "ll":
            def laxity(t):
                preds = list(G.predecessors(t))
                earliest = max(scheduled[p]["end_time"] for p in preds) if preds else 0
                return task_map[t]["deadline"] - task_map[t]["wcet"] - earliest
            ready_tasks.sort(key=laxity)

        for task_id in ready_tasks:
            task = task_map[task_id]
            preds = list(G.predecessors(task_id))
            ready_time = max(scheduled[p]["end_time"] for p in preds) if preds else 0

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
            break

    missed_deadlines = [t["task_id"] for t in schedule if t["end_time"] > t["deadline"]]
    return {
        "schedule": schedule,
        "missed_deadlines": missed_deadlines,
        "name": f"{policy.upper()} Multinode(without delay)"
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
